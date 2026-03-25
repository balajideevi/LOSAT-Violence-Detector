import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.video import r3d_18


class RWF2000Dataset(Dataset):
    CLASS_MAP = {"NonFight": 0, "Fight": 1, "NonViolence": 0, "Violence": 1}

    def __init__(self, root_dir: str, split: str = "val", clip_len: int = 16, size: int = 112):
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

        idx = torch.linspace(0, len(frames) - 1, steps=self.clip_len).long().tolist()
        return [frames[i] for i in idx]

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        clip = self._read_video(path)
        x = torch.from_numpy(np.stack(clip)).float() / 255.0
        x = x.permute(3, 0, 1, 2)
        mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
        std = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)
        x = (x - mean) / std
        return x, torch.tensor(label, dtype=torch.long), path


class ViolenceR3D18(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        try:
            from torchvision.models.video import R3D_18_Weights

            self.backbone = r3d_18(weights=R3D_18_Weights.DEFAULT)
        except ImportError:
            self.backbone = r3d_18(pretrained=True)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


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


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = RWF2000Dataset(root_dir=args.data_root, split=args.split, clip_len=16, size=112)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = ViolenceR3D18(num_classes=2).to(device)
    state_dict = torch.load(args.model_path, map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)
    model.eval()

    all_labels = []
    all_preds = []
    rows = []

    with torch.no_grad():
        for clips, labels, paths in loader:
            clips = clips.to(device)
            labels = labels.to(device)

            outputs = model(clips)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

            for path, label, pred, prob in zip(paths, labels.cpu().tolist(), preds.cpu().tolist(), probs[:, 1].cpu().tolist()):
                rows.append(
                    {
                        "path": path,
                        "actual": "Violence" if label == 1 else "Non-Violence",
                        "predicted": "Violence" if pred == 1 else "Non-Violence",
                        "violence_score": round(float(prob), 4),
                    }
                )

    metrics = compute_metrics(all_labels, all_preds)
    print("\nEvaluation Metrics")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"False Alarm Rate: {metrics['false_alarm_rate']:.4f}")
    print(f"Miss Rate: {metrics['miss_rate']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"TP: {metrics['tp']} TN: {metrics['tn']} FP: {metrics['fp']} FN: {metrics['fn']}")

    if args.output_csv:
        pd.DataFrame(rows).to_csv(args.output_csv, index=False)
        print(f"Saved per-video predictions to: {args.output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained R3D-18 on RWF-2000")
    parser.add_argument("--data_root", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--model_path", type=str, required=True, help="Path to best_model.pth")
    parser.add_argument("--split", type=str, default="val", help="Dataset split to evaluate")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output_csv", type=str, default="evaluation_predictions.csv")
    args = parser.parse_args()

    main(args)
