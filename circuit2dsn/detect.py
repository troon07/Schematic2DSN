"""Tiled component detection with per-component confidence.

A schematic page is often 3000-5000 px wide while a ground symbol or 0402
capacitor is ~20 px, so a single resized pass misses small parts. We run the
detector on the whole page (for large parts like ICs) and on overlapping
tiles at native resolution (for small parts), then merge with class-aware NMS.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# Confidence bands used by the review UI and the auto mode report.
HIGH, LOW = 0.80, 0.50


@dataclass
class Detection:
    id: str
    cls: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def band(self) -> str:
        return "high" if self.conf >= HIGH else "medium" if self.conf >= LOW else "low"


def load_pages(path: str | Path, dpi: int = 200) -> list[Image.Image]:
    """Return RGB page images from a PDF, or a single image file."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        import pymupdf

        pages = []
        with pymupdf.open(path) as doc:
            for page in doc:
                pix = page.get_pixmap(dpi=dpi, alpha=False)
                pages.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
        return pages
    return [Image.open(path).convert("RGB")]


def _origins(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    xs = list(range(0, length - tile, stride))
    xs.append(length - tile)
    return xs


class Detector:
    def __init__(self, weights: str | Path, imgsz: int = 1280, tile: int = 1280, overlap: float = 0.25,
                 conf: float = 0.10, iou: float = 0.5, device: str | None = None):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.names = self.model.names
        self.imgsz, self.tile, self.overlap = imgsz, tile, overlap
        self.conf, self.iou, self.device = conf, iou, device

    def _run(self, img: Image.Image, ox: int, oy: int) -> np.ndarray:
        r = self.model.predict(img, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                               device=self.device, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return np.zeros((0, 6), dtype=np.float32)
        b = r.boxes
        out = np.concatenate([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None], b.cls.cpu().numpy()[:, None]], 1)
        out[:, [0, 2]] += ox
        out[:, [1, 3]] += oy
        return out

    def detect(self, page: Image.Image) -> list[Detection]:
        import torch
        from torchvision.ops import batched_nms

        W, H = page.size
        parts = [self._run(page, 0, 0)]
        if max(W, H) > self.tile * 1.25:
            stride = int(self.tile * (1 - self.overlap))
            for oy in _origins(H, self.tile, stride):
                for ox in _origins(W, self.tile, stride):
                    crop = page.crop((ox, oy, min(ox + self.tile, W), min(oy + self.tile, H)))
                    parts.append(self._run(crop, ox, oy))
        dets = np.concatenate(parts, 0)
        if len(dets) == 0:
            return []
        t = torch.from_numpy(dets)
        keep = batched_nms(t[:, :4], t[:, 4], t[:, 5].long(), self.iou).numpy()
        dets = dets[keep]
        dets = dets[np.argsort(-dets[:, 4])]
        dets = _drop_contained(dets)
        return [Detection(f"C{i + 1}", self.names[int(d[5])], round(float(d[4]), 4), *map(float, d[:4]))
                for i, d in enumerate(dets)]


def _drop_contained(dets: np.ndarray, frac: float = 0.7) -> np.ndarray:
    """Drop a box that lies mostly (>= frac of its area) inside a more
    confident box: tile edges and partial views produce half-boxes of a part
    (e.g. a second "bjt" on one transistor's emitter). NMS misses these
    because the overlap is small relative to the big box."""
    keep = []
    for i, d in enumerate(dets):                      # sorted by confidence, high first
        area = max((d[2] - d[0]) * (d[3] - d[1]), 1e-6)
        inside = False
        for j in keep:
            k = dets[j]
            iw = max(0.0, min(d[2], k[2]) - max(d[0], k[0]))
            ih = max(0.0, min(d[3], k[3]) - max(d[1], k[1]))
            k_area = (k[2] - k[0]) * (k[3] - k[1])
            if iw * ih >= frac * area and k_area >= area:
                inside = True
                break
        if not inside:
            keep.append(i)
    return dets[keep]


def draw(page: Image.Image, dets: list[Detection]) -> Image.Image:
    from PIL import ImageDraw

    colours = {"high": (0, 170, 0), "medium": (255, 140, 0), "low": (220, 0, 0)}
    im = page.copy()
    d = ImageDraw.Draw(im)
    lw = max(2, round(max(im.size) / 1500))
    for det in dets:
        c = colours[det.band]
        d.rectangle([det.x1, det.y1, det.x2, det.y2], outline=c, width=lw)
        d.text((det.x1, det.y1 - 11), f"{det.id} {det.cls} {det.conf:.0%}", fill=c)
    return im


def save_json(path: Path, source: str, page_no: int, page: Image.Image, dets: list[Detection]) -> None:
    path.write_text(json.dumps({
        "source": source, "page": page_no, "width": page.size[0], "height": page.size[1],
        "thresholds": {"high": HIGH, "low": LOW},
        "components": [asdict(d) | {"band": d.band} for d in dets],
    }, indent=1), encoding="utf-8")
