from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import torch


def ensure_log_file(csv_path: str):
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        pd.DataFrame(columns=["timestamp", "score", "threshold", "motion", "decision"]).to_csv(path, index=False)


def log_event(csv_path: str, score: float, threshold: float, motion: float, decision: str):
    ensure_log_file(csv_path)
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "score": round(float(score), 4),
        "threshold": round(float(threshold), 4),
        "motion": round(float(motion), 4),
        "decision": decision,
    }
    pd.DataFrame([row]).to_csv(csv_path, mode="a", header=False, index=False)


def read_event_log(csv_path: str, tail: int = 30) -> pd.DataFrame:
    ensure_log_file(csv_path)
    df = pd.read_csv(csv_path)
    return df.tail(tail)


def preprocess_clip(frames_rgb: list[np.ndarray], size: int = 112) -> torch.Tensor:
    processed = []
    for frame in frames_rgb:
        frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
        processed.append(frame)

    clip = np.stack(processed).astype(np.float32) / 255.0  # [T,H,W,C]
    clip = torch.from_numpy(clip).permute(3, 0, 1, 2)  # [C,T,H,W]

    # Kinetics normalization for R3D models.
    mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
    std = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)
    clip = (clip - mean) / std

    return clip.unsqueeze(0)  # [1,C,T,H,W]


def compute_motion_metric(frames_rgb: list[np.ndarray]) -> float:
    if len(frames_rgb) < 2:
        return 0.0

    diffs = []
    for i in range(1, len(frames_rgb)):
        a = frames_rgb[i - 1].astype(np.float32) / 255.0
        b = frames_rgb[i].astype(np.float32) / 255.0
        diffs.append(np.mean(np.abs(b - a)))

    return float(np.mean(diffs))


def add_alert_border(frame_bgr: np.ndarray, alert: bool, thickness: int = 8) -> np.ndarray:
    if not alert:
        return frame_bgr

    frame = frame_bgr.copy()
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), thickness)
    return frame
