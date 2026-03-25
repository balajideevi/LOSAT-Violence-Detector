from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class RWF2000Dataset(Dataset):
    """
    Expects one of these structures:
    1) root/train/Fight, root/train/NonFight, root/val/Fight, root/val/NonFight
    2) root/Fight, root/NonFight (for custom split outside this class)
    """

    CLASS_MAP = {"NonFight": 0, "Fight": 1, "NonViolence": 0, "Violence": 1}

    def __init__(self, root_dir: str, split: str | None = None, clip_len: int = 16, size: int = 112):
        self.root_dir = Path(root_dir)
        self.split = split
        self.clip_len = clip_len
        self.size = size
        self.samples = self._gather_samples()

        if not self.samples:
            raise ValueError(f"No video samples found in: {self.root_dir}")

    def _gather_samples(self):
        base = self.root_dir / self.split if self.split else self.root_dir
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
            # Fallback black clip.
            black = np.zeros((self.size, self.size, 3), dtype=np.uint8)
            frames = [black.copy() for _ in range(self.clip_len)]

        # Uniform temporal sampling to fixed clip length.
        indices = torch.linspace(0, len(frames) - 1, steps=self.clip_len).long().tolist()
        clip = [frames[i] for i in indices]
        return clip

    def _to_tensor(self, clip):
        clip_tensor = torch.from_numpy(np.stack(clip)).float() / 255.0  # [T,H,W,C]
        clip_tensor = clip_tensor.permute(3, 0, 1, 2)  # [C,T,H,W]

        mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
        std = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)
        clip_tensor = (clip_tensor - mean) / std
        return clip_tensor

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        clip = self._read_video(path)
        clip_tensor = self._to_tensor(clip)
        return clip_tensor, torch.tensor(label, dtype=torch.long)
