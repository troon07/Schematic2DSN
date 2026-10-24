"""Image2Net orientation crops -> two ultralytics classification datasets.

    python scripts/prepare_orientation_data.py

data/cls/orient:  diode_{u,d,l,r}, bjt_{u,d,l,r}, mos_{u,d,l,r}
                  diode: the way the triangle points (cathode side)
                  bjt:   the side the base is on
                  mos:   the side the gate is on
data/cls/mirror:  bjt_m0 / bjt_m1, crops drawn base-left:
                  m0 = emitter arrow on the upper leg, m1 = on the lower leg

Rotated copies of one drawing share a number (u12, d12, l12, r12), so the
train/val split is by that number: copies never land on both sides.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "raw" / "ci2n_datasets-main" / "device_orientation"
ORIENT = {"train_diode": "diode", "train_bjt_d": "bjt", "train_mos": "mos"}
MIRROR = {"train_bjt_m": "bjt"}


def is_val(name: str, every: int) -> bool:
    m = re.search(r"(\d+)", name)
    return bool(m) and int(m.group(1)) % every == 0


def build(out: Path, mapping: dict, every: int, mirror: bool) -> dict:
    if out.exists():
        shutil.rmtree(out)
    counts = {}
    for folder, dev in mapping.items():
        for cls_dir in sorted((SRC / folder).iterdir()):
            if not cls_dir.is_dir():
                continue
            label = f"{dev}_m{cls_dir.name}" if mirror else f"{dev}_{cls_dir.name}"
            for f in cls_dir.iterdir():
                if f.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                split = "val" if is_val(f.stem, every) else "train"
                dst = out / split / label
                dst.mkdir(parents=True, exist_ok=True)
                shutil.copy(f, dst / f"{folder}_{f.name}")
                counts[(split, label)] = counts.get((split, label), 0) + 1
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--every", type=int, default=7, help="drawing numbers divisible by this go to val")
    args = ap.parse_args()
    if not SRC.exists():
        raise SystemExit(f"missing {SRC}: unzip ci2n_datasets-main.zip (device_orientation) into data/raw")
    for name, mapping, mirror in (("orient", ORIENT, False), ("mirror", MIRROR, True)):
        counts = build(ROOT / "data" / "cls" / name, mapping, args.every, mirror)
        labels = sorted({k[1] for k in counts})
        print(f"data/cls/{name}: " + ", ".join(f"{l} {counts.get(('train', l), 0)}/{counts.get(('val', l), 0)}" for l in labels))


if __name__ == "__main__":
    main()
