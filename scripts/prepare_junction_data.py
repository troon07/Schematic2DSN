"""Patches around wire junctions, labelled from the Image2Net crossing dataset.

    python scripts/prepare_junction_data.py            # -> data/junction/{train,val}.npz

jumper_identification has 28,195 labelled crossings in many drawing styles:
class 0 = hop arc, 1 = junction DOT (wires connected), 2 = junction with no dot.
For every image we trace the wires exactly as the pipeline does and take each
skeleton junction (3+ arms) that falls inside a labelled box. The patch is the
grey image around the junction, 10 line-widths wide, resized to 32 x 32, so a
tiny dot on a blurry 540 px page and a big dot on a 1800 px page look alike.

Saved per patch: label (0 hop / 1 dot / 2 plain), the number of arms, the old
thickness-rule decision (for comparison) and the image name.
"""

from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from circuit2dsn.wires import build_graph, ink_mask  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "raw" / "ci2n_datasets-main" / "jumper_identification"
PATCH = 32


def patch_at(gray: np.ndarray, x: float, y: float, width: float) -> np.ndarray:
    """32 x 32 crop centred on (x, y), 10 line widths wide, contrast-normalised."""
    half = max(5.0 * width, 8.0)
    x0, y0 = x - half, y - half
    scale = PATCH / (2 * half)
    M = np.array([[scale, 0, -x0 * scale], [0, scale, -y0 * scale]], dtype=np.float32)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    p = cv2.warpAffine(gray, M, (PATCH, PATCH), flags=interp, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    bg = np.percentile(p, 90)                      # paper level
    ink = np.percentile(p, 2)
    p = (bg - p) / max(bg - ink, 30.0)             # 0 = paper, 1 = darkest ink
    return np.clip(p * 255, 0, 255).astype(np.uint8)


def one(args):
    img, lab, name = args
    rgb = np.asarray(Image.open(img).convert("RGB"))
    H, W = rgb.shape[:2]
    gray = rgb.min(2) if rgb.ndim == 3 else rgb
    boxes = []
    for line in Path(lab).read_text().splitlines():
        r = line.split()
        if len(r) < 9:
            continue
        xs = [float(r[1 + 2 * i]) * W for i in range(4)]
        ys = [float(r[2 + 2 * i]) * H for i in range(4)]
        boxes.append((int(r[0]), min(xs), min(ys), max(xs), max(ys)))
    if not boxes:
        return []
    g = build_graph(ink_mask(rgb))
    width = max(2.0, 2 * g.stroke)
    out = []
    for j in g.junctions:
        if len(j.arms) < 3:
            continue
        hits = [(abs(j.x - (b[1] + b[3]) / 2) + abs(j.y - (b[2] + b[4]) / 2), b[0]) for b in boxes
                if b[1] - 3 <= j.x <= b[3] + 3 and b[2] - 3 <= j.y <= b[4] + 3]
        if not hits:
            continue
        label = min(hits)[1]
        out.append((patch_at(gray, j.x, j.y, width), label, len(j.arms), 1 if j.kind == "dot" else 0, name))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "junction")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        jobs = []
        for img in sorted((SRC / "images" / split).glob("*.*")):
            lab = SRC / "labels" / split / (img.stem + ".txt")
            if lab.exists():
                jobs.append((str(img), str(lab), img.stem))
        rows = []
        with Pool(args.workers) as pool:
            for k, r in enumerate(pool.imap_unordered(one, jobs, chunksize=2), 1):
                rows += r
                if k % 100 == 0:
                    print(f"  {split}: {k}/{len(jobs)} images, {len(rows)} junctions", flush=True)
        if not rows:
            continue
        P = np.stack([r[0] for r in rows])
        np.savez_compressed(args.out / f"{split}.npz", patches=P, label=np.array([r[1] for r in rows]),
                            arms=np.array([r[2] for r in rows]), old_dot=np.array([r[3] for r in rows]),
                            image=np.array([r[4] for r in rows]))
        lab = np.array([r[1] for r in rows])
        print(f"{split}: {len(rows)} junctions  hop {int((lab == 0).sum())}  dot {int((lab == 1).sum())}  "
              f"plain {int((lab == 2).sum())}")


if __name__ == "__main__":
    main()
