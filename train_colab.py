import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision.models.video import r3d_18


class RWF2000Dataset(Dataset):
    """
    Expected structure:
    /data_root/train/Fight, /data_root/train/NonFight
    /data_root/val/Fight,   /data_root/val/NonFight
    """

    CLASS_MAP = {"NonFight": 0, "Fight": 1, "NonViolence": 0, "Violence": 1}

    def __init__(self, root_dir: str, split: str = "train", clip_len: int = 16, size: int = 112):
        self.root_dir = Path(root_dir)
        self.split = split
        self.clip_len = clip_len
        self.size = size
        self.samples = self._gather_samples()

        if not self.samples:
            raise ValueError(f"No video samples found in: {self.root_dir / self.split}")

    def _gather_samples(self):
        base = self.root_dir / self.split
        if not base.exists():
            raise ValueError(f"Split path not found: {base}")

        samples = []
        for class_name, label in self.CLASS_MAP.items():
            class_dir = base / class_name
            if not class_dir.exists():
                continue
            for ext in ("*.avi", "*.mp4", "*.mov", "*.mkv"):
                for path in class_dir.glob(ext):
                    samples.append((str(path), label))
        return samples

    def __len__(self):
        return len(self.samples)

    def _read_video(self, path: str):
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
            frames.append(frame)
        cap.release()

        if len(frames) == 0:
            black = np.zeros((self.size, self.size, 3), dtype=np.uint8)
            frames = [black.copy() for _ in range(self.clip_len)]

        # Uniform temporal sampling to fixed clip length.
        idx = torch.linspace(0, len(frames) - 1, steps=self.clip_len).long().tolist()
        return [frames[i] for i in idx]

    def _to_tensor(self, clip):
        x = torch.from_numpy(np.stack(clip)).float() / 255.0  # [T,H,W,C]
        x = x.permute(3, 0, 1, 2)  # [C,T,H,W]
        mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
        std = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)
        return (x - mean) / std

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        clip = self._read_video(path)
        return self._to_tensor(clip), torch.tensor(label, dtype=torch.long)


class ViolenceR3D18(nn.Module):
    """R3D-18 with 2-class head (Violence / Non-Violence)."""

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        try:
            self.backbone = r3d_18(pretrained=pretrained)
        except TypeError:
            from torchvision.models.video import R3D_18_Weights

            weights = R3D_18_Weights.DEFAULT if pretrained else None
            self.backbone = r3d_18(weights=weights)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for clips, labels in loader:
        clips = clips.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(clips)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * clips.size(0)
        preds = outputs.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        total += clips.size(0)

    return running_loss / total, running_correct / total


def compute_metrics(y_true, y_pred):
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    total = max(1, len(y_true))
    accuracy = (tp + tn) / total
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    false_alarm_rate = fp / max(1, fp + tn)
    miss_rate = fn / max(1, fn + tp)
    f1_score = 2 * precision * recall / max(1e-8, precision + recall)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "false_alarm_rate": false_alarm_rate,
        "miss_rate": miss_rate,
        "f1_score": f1_score,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for clips, labels in loader:
            clips = clips.to(device)
            labels = labels.to(device)

            outputs = model(clips)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * clips.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            total += clips.size(0)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    metrics = compute_metrics(all_labels, all_preds)
    return running_loss / total, running_correct / total, metrics


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = RWF2000Dataset(root_dir=args.data_root, split="train", clip_len=16, size=112)
    val_ds = RWF2000Dataset(root_dir=args.data_root, split="val", clip_len=16, size=112)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = ViolenceR3D18(num_classes=2, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.output_dir, exist_ok=True)
    best_acc = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_metrics = validate(model, val_loader, criterion, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_accuracy": round(train_acc, 4),
                "val_loss": round(val_loss, 4),
                "val_accuracy": round(val_metrics["accuracy"], 4),
                "precision": round(val_metrics["precision"], 4),
                "recall": round(val_metrics["recall"], 4),
                "specificity": round(val_metrics["specificity"], 4),
                "false_alarm_rate": round(val_metrics["false_alarm_rate"], 4),
                "miss_rate": round(val_metrics["miss_rate"], 4),
                "f1_score": round(val_metrics["f1_score"], 4),
                "tp": val_metrics["tp"],
                "tn": val_metrics["tn"],
                "fp": val_metrics["fp"],
                "fn": val_metrics["fn"],
            }
        )

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, "
            f"F1: {val_metrics['f1_score']:.4f}, FAR: {val_metrics['false_alarm_rate']:.4f}"
        )
        print(
            f"Confusion Matrix | TP: {val_metrics['tp']} TN: {val_metrics['tn']} "
            f"FP: {val_metrics['fp']} FN: {val_metrics['fn']}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model to: {best_path}")

    final_path = os.path.join(args.output_dir, "final_model.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Saved final model to: {final_path}")

    metrics_path = os.path.join(args.output_dir, "validation_metrics.csv")
    pd.DataFrame(history).to_csv(metrics_path, index=False)
    print(f"Saved validation metrics to: {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train R3D-18 on RWF-2000 (Colab)")
    parser.add_argument("--data_root", type=str, required=True, help="Path to RWF-2000 root")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    main(args)
