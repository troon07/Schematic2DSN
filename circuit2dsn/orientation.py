"""Which pin is which on polarised parts: diode/LED anode+cathode, BJT
base/collector/emitter, MOSFET gate.

Two small classifiers (scripts/train_orientation.py, Image2Net crops):
- orient: diode_{u,d,l,r} = way the triangle points (the cathode side);
          bjt_{..} = side of the base; mos_{..} = side of the gate
- mirror: on a BJT crop rotated base-left, m0 = emitter on the upper leg,
          m1 = emitter on the lower leg
Pin roles then follow from pin positions relative to the part's centre.
Standard pin numbers (KiCad Device library order) are used only when the
drawing's own pin numbers were not read.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import PROJECT_ROOT

DEVICE = {"diode": "diode", "led": "diode", "bjt": "bjt", "mosfet": "mos"}
DIRS = {"u": (0.0, -1.0), "d": (0.0, 1.0), "l": (-1.0, 0.0), "r": (1.0, 0.0)}
# KiCad Device:D (1=K, 2=A), Q_NPN_BCE (1=B, 2=C, 3=E), Q_NMOS_GDS (1=G, 2=D, 3=S)
STANDARD_NUMBER = {"K": "1", "A": "2", "B": "1", "C": "2", "E": "3", "G": "1"}
DEFAULT_ORIENT = PROJECT_ROOT / "runs" / "orient" / "weights" / "best.pt"
DEFAULT_MIRROR = PROJECT_ROOT / "runs" / "mirror" / "weights" / "best.pt"


def _to_base_left(img: np.ndarray, side: str) -> np.ndarray:
    return {"l": img, "r": cv2.rotate(img, cv2.ROTATE_180),
            "u": cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE),
            "d": cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)}[side]


def _vec_to_base_left(v: np.ndarray, side: str) -> np.ndarray:
    """The same rotation as _to_base_left, applied to (x, y) offsets (y down)."""
    x, y = v
    return {"l": np.array([x, y]), "r": np.array([-x, -y]),
            "u": np.array([y, -x]), "d": np.array([-y, x])}[side]


def pin_roles(cls: str, box, pins: dict[str, tuple[float, float]], side: str, mirror: str | None) -> dict[str, str]:
    """pins: {pid: (x, y)}. side: u/d/l/r from the orient model."""
    if not pins or side not in DIRS:
        return {}
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    vec = {pid: np.array([x - cx, y - cy]) for pid, (x, y) in pins.items()}
    d = np.array(DIRS[side])
    toward = max(vec, key=lambda p: float(vec[p] @ d))
    dev = DEVICE.get(cls)
    if dev == "diode":
        if len(pins) != 2:
            return {}
        return {p: ("K" if p == toward else "A") for p in pins}
    if dev == "mos":
        return {toward: "G"}
    if dev == "bjt":
        roles = {toward: "B"}
        rest = [p for p in pins if p != toward]
        if len(rest) == 2 and mirror in ("0", "1"):
            upper = min(rest, key=lambda p: float(_vec_to_base_left(vec[p], side)[1]))
            lower = rest[0] if rest[1] == upper else rest[1]
            emitter = upper if mirror == "0" else lower
            for p in rest:
                roles[p] = "E" if p == emitter else "C"
        return roles
    return {}


class Orienter:
    def __init__(self, orient_weights: Path = DEFAULT_ORIENT, mirror_weights: Path | None = DEFAULT_MIRROR):
        from ultralytics import YOLO

        self.orient = YOLO(str(orient_weights))
        self.mirror = YOLO(str(mirror_weights)) if mirror_weights and Path(mirror_weights).exists() else None

    @classmethod
    def available(cls) -> "Orienter | None":
        return cls() if DEFAULT_ORIENT.exists() else None

    def _probs(self, model, img: np.ndarray) -> dict[str, float]:
        r = model.predict(img, imgsz=64, verbose=False)[0]
        return {model.names[i]: float(p) for i, p in enumerate(r.probs.data.cpu().numpy())}

    def classify(self, rgb: np.ndarray, cls: str, box) -> tuple[str | None, str | None, float]:
        """(side, mirror, confidence) for one part; side None if not a polarised class."""
        dev = DEVICE.get(cls)
        if dev is None:
            return None, None, 0.0
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        crop = rgb[max(y1, 0):y2 + 1, max(x1, 0):x2 + 1]
        if crop.size == 0:
            return None, None, 0.0
        crop = np.ascontiguousarray(crop[:, :, ::-1])            # ultralytics expects BGR arrays
        probs = {k: v for k, v in self._probs(self.orient, crop).items() if k.startswith(dev + "_")}
        if not probs:
            return None, None, 0.0
        label = max(probs, key=probs.get)
        total = sum(probs.values()) or 1.0
        side, conf = label.split("_")[1], probs[label] / total
        mirror = None
        if dev == "bjt" and self.mirror is not None:
            mp = self._probs(self.mirror, np.ascontiguousarray(_to_base_left(crop, side)))
            m = max(mp, key=mp.get)
            mirror, conf = m.split("_m")[1], conf * mp[m]
        return side, mirror, round(conf, 3)
