"""Add your reviewed pages to the component-detector training data.

Every review export writes outputs/review/<name>/training/ (the page plus
corrected YOLO labels). This script collects them all and builds
data/yolo/detector_plus: the original dataset plus your pages, tiled the
same way and repeated --repeat times (they are the drawing styles you
actually use, so they should weigh more than one page among thousands).
Every 5th reviewed page goes to validation so the gain can be measured.

    python scripts/prepare_retrain.py
    then fine-tune (notebook Part 3, or):
    python scripts/train_detector.py --data data/yolo/detector_plus/dataset.yaml \
        --model runs/detector/weights/best.pt --epochs 30 --name detector_ft
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from circuit2dsn.taxonomy import Taxonomy  # noqa: E402
from scripts.prepare_detector_data import Box, Sample, export, yaml_dump  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
Image.MAX_IMAGE_PIXELS = None


def held_out_pcbnet() -> set[str]:
    """PCBnet val/test designs: reviewing them (demos) must not leak into training."""
    import json

    path = ROOT / "data" / "yolo" / "detector" / "splits.json"
    if not path.exists():
        return set()
    sp = json.loads(path.read_text())["pcbnet"]
    return {u.removeprefix("pcbnet_") for k in ("val", "test") for u in sp[k]}


def reviewed_samples(review_root: Path, tax: Taxonomy) -> list[Sample]:
    out = []
    held = held_out_pcbnet()
    for lbl in sorted(review_root.glob("*/training/labels/*.txt")):
        img = lbl.parent.parent / "images" / (lbl.stem + ".png")
        if not img.exists():
            continue
        m = lbl.stem.split("_")
        if lbl.stem.startswith("pcbnet_") and len(m) > 1 and m[1] in held:
            print(f"  skip {lbl.stem}: PCBnet val/test design (would leak into training)")
            continue
        W, H = Image.open(img).size
        s = Sample(f"review_{lbl.stem}", "review", img)
        for line in lbl.read_text(encoding="utf-8").split("\n"):
            if not line.strip():
                continue
            c, cx, cy, w, h = line.split()
            cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
            s.boxes.append(Box(tax.classes[int(c)], None, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, "review"))
        out.append(s)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, default=ROOT / "data" / "yolo" / "detector")
    ap.add_argument("--reviews", type=Path, default=ROOT / "outputs" / "review")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "yolo" / "detector_plus")
    ap.add_argument("--repeat", type=int, default=5, help="copies of each reviewed training page")
    ap.add_argument("--tile", type=int, default=1280)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--tile-trigger", type=float, default=1.25)
    ap.add_argument("--min-visible", type=float, default=0.6)
    ap.add_argument("--keep-empty", type=float, default=0.15)
    ap.add_argument("--zip", type=Path, help="only pack outputs/review/*/training into this zip (for Colab) and stop")
    args = ap.parse_args()

    if args.zip:
        import zipfile

        files = [f for f in args.reviews.glob("*/training/**/*") if f.is_file()]
        with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as z:
            for f in files:
                z.write(f, f.relative_to(args.reviews.parent.parent))
        print(f"wrote {args.zip} ({len(files)} files): upload it to Drive T2/ for notebook Part 3")
        return

    tax = Taxonomy.load()
    if not (args.base / "dataset.yaml").exists():
        sys.exit(f"base dataset missing: run scripts/prepare_detector_data.py first ({args.base})")
    samples = reviewed_samples(args.reviews, tax)
    if not samples:
        sys.exit(f"no reviewed pages found under {args.reviews} (export a review first)")

    if args.out.exists():
        shutil.rmtree(args.out)
    print(f"copying base dataset {args.base} ...")
    for sub in ("images", "labels"):
        shutil.copytree(args.base / sub, args.out / sub)

    added = {"train": 0, "val": 0}
    for i, s in enumerate(samples):
        split = "val" if i % 5 == 4 else "train"
        copies = args.repeat if split == "train" else 1
        for k in range(copies):
            uid = s.uid if k == 0 else f"{s.uid}_r{k}"
            added[split] += export(Sample(uid, s.source, s.image, s.boxes), split, args.out, tax, args)
    (args.out / "dataset.yaml").write_text(yaml_dump(args.out, tax.classes), encoding="utf-8")
    print(f"{len(samples)} reviewed pages -> {added['train']} train images (x{args.repeat}), {added['val']} val images")
    print(f"dataset: {args.out / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
