"""Train the YOLO component detector.

Local CPU smoke test (a few minutes, only checks that the pipeline runs):
    python scripts/train_detector.py --smoke

Real training (needs an NVIDIA GPU, e.g. Colab/Kaggle; see notebooks/):
    python scripts/train_detector.py --model yolo11s.pt --epochs 150 --imgsz 1280 --batch 8

Augmentation notes: schematics are line art, so colour jitter is mild,
mosaic helps with dense layouts, and flips are OFF. A mirrored diode or
transistor is a different part orientation, and the later orientation /
pin-mapping stage relies on the detector not being trained to ignore
mirroring.
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "yolo" / "detector" / "dataset.yaml")
    ap.add_argument("--model", default="yolo11s.pt", help="pretrained weights or a .yaml to train from scratch")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None, help="'0' for first GPU, 'cpu' to force CPU")
    ap.add_argument("--name", default="detector")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--time", type=float, default=None, help="hard time budget in hours (schedule fits it)")
    ap.add_argument("--close-mosaic", type=int, default=15, help="last N epochs without mosaic (use ~2 for short runs)")
    ap.add_argument("--smoke", action="store_true", help="tiny CPU run to verify the setup")
    args = ap.parse_args()

    from ultralytics import YOLO

    if args.smoke:
        args.model, args.epochs, args.imgsz, args.batch, args.device = "yolo11n.pt", 1, 640, 4, "cpu"
        args.name = "detector_smoke"

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(ROOT / "runs"),
        name=args.name,
        exist_ok=args.smoke,
        resume=args.resume,
        patience=30,
        cos_lr=True,
        close_mosaic=args.close_mosaic,
        fliplr=0.0,
        flipud=0.0,
        degrees=0.0,
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.3,
        scale=0.4,
        translate=0.1,
        mosaic=1.0,
        workers=2 if args.smoke else 8,
        fraction=0.05 if args.smoke else 1.0,
        plots=True,
        seed=42,
        time=args.time,
    )
    metrics = model.val(split="val", imgsz=args.imgsz, batch=args.batch, device=args.device,
                        project=str(ROOT / "runs"), name=f"{args.name}_val", exist_ok=True)
    print(f"\nval mAP50-95={metrics.box.map:.4f}  mAP50={metrics.box.map50:.4f}")
    print(f"best weights: {Path(model.trainer.best)}")


if __name__ == "__main__":
    main()
