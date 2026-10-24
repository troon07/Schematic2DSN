"""Train the two small orientation classifiers (CPU is fine: 64 px crops).

    python scripts/train_orientation.py              # both models
    python scripts/train_orientation.py --task mirror

Flips and rotations are OFF: a flipped left-facing diode is a right-facing
diode, so those augmentations would teach the wrong label.
Weights: runs/orient/weights/best.pt, runs/mirror/weights/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["orient", "mirror", "both"], default="both")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--imgsz", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from ultralytics import YOLO

    for task in (["orient", "mirror"] if args.task == "both" else [args.task]):
        model = YOLO("yolo11n-cls.pt")
        model.train(data=str(ROOT / "data" / "cls" / task), epochs=args.epochs, imgsz=args.imgsz, batch=64,
                    device=args.device, project=str(ROOT / "runs"), name=task, exist_ok=True,
                    fliplr=0.0, flipud=0.0, degrees=0.0, auto_augment=None, erasing=0.0,
                    hsv_h=0.0, hsv_s=0.3, hsv_v=0.3, scale=0.2, translate=0.1,
                    workers=2, patience=8, seed=42, plots=False)
        m = model.val(data=str(ROOT / "data" / "cls" / task), imgsz=args.imgsz, device=args.device, plots=False)
        print(f"{task}: val top-1 = {m.top1:.3f}")


if __name__ == "__main__":
    main()
