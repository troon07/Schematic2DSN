"""Detect components in a schematic PDF or image and report confidence.

    python scripts/detect.py my_circuit.pdf --weights runs/detector/weights/best.pt

Writes, per page, into --out:
    <name>_p<N>.json   every component with class, box, confidence and band
    <name>_p<N>.png    overlay: green >= 80%, orange 50-80%, red < 50%
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from circuit2dsn.detect import Detector, draw, load_pages, save_json  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="PDF, PNG or JPG")
    ap.add_argument("--weights", type=Path, default=ROOT / "runs" / "detector" / "weights" / "best.pt")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "detect")
    ap.add_argument("--dpi", type=int, default=200, help="PDF render resolution")
    ap.add_argument("--conf", type=float, default=0.10, help="lowest confidence to report")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    if not args.input.is_file():
        sys.exit(f"input not found: {args.input}")
    if not args.weights.exists():
        sys.exit(f"weights not found: {args.weights}\nTrain first: python scripts/train_detector.py")
    args.out.mkdir(parents=True, exist_ok=True)
    det = Detector(args.weights, imgsz=args.imgsz, tile=args.imgsz, conf=args.conf, device=args.device)

    for n, page in enumerate(load_pages(args.input, args.dpi), 1):
        dets = det.detect(page)
        stem = f"{args.input.stem}_p{n}"
        save_json(args.out / f"{stem}.json", str(args.input), n, page, dets)
        draw(page, dets).save(args.out / f"{stem}.png")
        bands = collections.Counter(d.band for d in dets)
        classes = collections.Counter(d.cls for d in dets)
        print(f"page {n}: {len(dets)} components  high={bands['high']} medium={bands['medium']} low={bands['low']}")
        print("   " + ", ".join(f"{c}x{k}" for k, c in classes.most_common()))
        print(f"   -> {args.out / stem}.png")


if __name__ == "__main__":
    main()
