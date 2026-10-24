"""Junction classifier: is a 4-way wire junction a dot (connected) or not?

JunctionNet is a tiny CNN on 32 x 32 grey patches (see scripts/prepare_junction_data.py,
scripts/train_junction.py). `p_dot(gray, x, y, width)` returns the probability that
the junction at (x, y) is a connection dot, or None when no trained model is
installed (the old thickness rule is then used alone).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

MODEL = Path(__file__).with_name("models") / "junction.pt"
PATCH = 32


def _net():
    import torch.nn as nn

    class JunctionNet(nn.Module):
        def __init__(self):
            super().__init__()

            def block(i, o):
                return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

            self.f = nn.Sequential(block(1, 16), block(16, 16), nn.MaxPool2d(2),
                                   block(16, 32), block(32, 32), nn.MaxPool2d(2),
                                   block(32, 64), nn.MaxPool2d(2))
            self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(64 * 16, 64), nn.ReLU(inplace=True),
                                      nn.Linear(64, 3))

        def forward(self, x):
            return self.head(self.f(x))

    return JunctionNet


def __getattr__(name):                      # `from circuit2dsn.junction_model import JunctionNet`
    if name == "JunctionNet":
        return _net()
    raise AttributeError(name)


@lru_cache(maxsize=1)
def _loaded():
    if not MODEL.exists():
        return None
    import torch

    ck = torch.load(MODEL, map_location="cpu", weights_only=False)
    m = _net()()
    m.load_state_dict(ck["state"])
    m.eval()
    return m


def available() -> bool:
    return MODEL.exists()


def patch_at(gray: np.ndarray, x: float, y: float, width: float) -> np.ndarray:
    import cv2

    half = max(5.0 * width, 8.0)
    scale = PATCH / (2 * half)
    M = np.array([[scale, 0, -(x - half) * scale], [0, scale, -(y - half) * scale]], dtype=np.float32)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    p = cv2.warpAffine(gray, M, (PATCH, PATCH), flags=interp, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    bg, ink = np.percentile(p, 90), np.percentile(p, 2)
    p = (bg - p) / max(bg - ink, 30.0)
    return np.clip(p * 255, 0, 255).astype(np.uint8)


def p_dot_many(gray: np.ndarray, points: list[tuple[float, float, float]]) -> list[float] | None:
    """Dot probability for several junctions (x, y, line width) in one pass."""
    m = _loaded()
    if m is None or not points:
        return None
    import torch

    X = torch.from_numpy(np.stack([patch_at(gray, x, y, w) for x, y, w in points])).float().div(255).unsqueeze(1)
    with torch.no_grad():
        p = torch.softmax(m(X), 1)[:, 1]
    return [float(v) for v in p]
