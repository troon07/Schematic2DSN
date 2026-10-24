"""Convert PCBnet (Dataset300) and Image2Net (ci2n) into one YOLO detection dataset.

Output layout (under --out):
    images/{train,val,test}/*.png
    labels/{train,val,test}/*.txt      YOLO: cls cx cy w h (normalised)
    dataset.yaml                       for ultralytics
    annotations.jsonl                  every source image with full-res boxes + subtype
    splits.json                        which source image went to which split

Large schematics are cut into overlapping tiles for train/val so small parts
(0402 caps, ground symbols) keep enough pixels; the downscaled full image is
kept too so the model also sees whole-page context. Test images stay whole:
evaluation and inference run the tiled predictor on the original page.

Usage:
    python scripts/prepare_detector_data.py
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from circuit2dsn.taxonomy import Taxonomy  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
IMAGE_EXTS = (".png", ".jpg", ".jpeg")


@dataclass
class Box:
    cls: str
    subtype: str | None
    x1: float
    y1: float
    x2: float
    y2: float
    raw: str


@dataclass
class Sample:
    uid: str
    source: str
    image: Path
    boxes: list[Box] = field(default_factory=list)


def load_pcbnet(root: Path, tax: Taxonomy, unmapped: collections.Counter) -> list[Sample]:
    samples = []
    subs = [p for p in root.iterdir() if p.is_dir()]
    while len(subs) == 1 and not list(subs[0].glob("*_CVJsonStd.json")):   # unpacked one level deeper
        root = subs[0]
        subs = [p for p in root.iterdir() if p.is_dir()]
    for d in sorted(subs):
        imgs = [f for f in d.iterdir() if f.suffix.lower() in IMAGE_EXTS]
        jsons = list(d.glob("*_CVJsonStd.json"))
        if len(imgs) != 1 or len(jsons) != 1:
            print(f"  skip {d.name}: expected 1 image + 1 json", file=sys.stderr)
            continue
        data = json.loads(jsons[0].read_text(encoding="utf-8"))
        s = Sample(f"pcbnet_{d.name}", "pcbnet", imgs[0])
        for comp in data["components"].values():
            if comp.get("state") not in (None, "active"):
                continue
            m = tax.map("pcbnet", comp["category"])
            if m is None:
                unmapped[f"pcbnet:{comp['category']}"] += 1
                continue
            if m[0] is None:
                continue
            # `location` includes pin stubs; that is what we want, pins are
            # where wires attach.
            loc = comp["location"]
            s.boxes.append(Box(m[0], m[1], loc["x1"], loc["y1"], loc["x2"], loc["y2"], comp["category"]))
        samples.append(s)
    return samples


def load_ci2n(root: Path, tax: Taxonomy, unmapped: collections.Counter) -> list[Sample]:
    samples = []
    for jf in sorted(root.glob("*.json"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem):
        img = next((jf.with_suffix(e) for e in IMAGE_EXTS if jf.with_suffix(e).exists()), None)
        if img is None:
            continue
        data = json.loads(jf.read_text(encoding="utf-8"))
        s = Sample(f"ci2n_{jf.stem}", "ci2n", img)
        for shape in data["shapes"]:
            if shape.get("shape_type") != "rectangle":
                continue
            m = tax.map("ci2n", shape["label"])
            if m is None:
                unmapped[f"ci2n:{shape['label']}"] += 1
                continue
            if m[0] is None:
                continue
            (ax, ay), (bx, by) = shape["points"][:2]
            s.boxes.append(Box(m[0], m[1], min(ax, bx), min(ay, by), max(ax, bx), max(ay, by), shape["label"]))
        samples.append(s)
    return samples


def split_samples(samples: list[Sample], ratios: tuple[float, float, float], rng: random.Random) -> dict[str, list[Sample]]:
    items = samples[:]
    rng.shuffle(items)
    n = len(items)
    n_val = round(n * ratios[1])
    n_test = round(n * ratios[2])
    return {
        "test": items[:n_test],
        "val": items[n_test:n_test + n_val],
        "train": items[n_test + n_val:],
    }


def tile_origins(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    xs = list(range(0, length - tile, stride))
    xs.append(length - tile)
    return xs


def yolo_lines(boxes: list[Box], ox: int, oy: int, w: int, h: int, tax: Taxonomy, min_visible: float) -> list[str]:
    lines = []
    for b in boxes:
        x1, y1 = max(b.x1 - ox, 0), max(b.y1 - oy, 0)
        x2, y2 = min(b.x2 - ox, w), min(b.y2 - oy, h)
        if x2 <= x1 or y2 <= y1:
            continue
        full = max((b.x2 - b.x1) * (b.y2 - b.y1), 1e-6)
        if (x2 - x1) * (y2 - y1) / full < min_visible:
            continue
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        lines.append(f"{tax.index(b.cls)} {cx:.6f} {cy:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    return lines


def write_image(img: Image.Image, lines: list[str], out: Path, split: str, name: str) -> None:
    img.save(out / "images" / split / f"{name}.png", optimize=False)
    (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def export(sample: Sample, split: str, out: Path, tax: Taxonomy, args) -> int:
    img = Image.open(sample.image).convert("RGB")
    W, H = img.size
    written = 0
    # Whole page (ultralytics resizes it to imgsz).
    write_image(img, yolo_lines(sample.boxes, 0, 0, W, H, tax, 1.0), out, split, sample.uid)
    written += 1
    if split == "test" or max(W, H) <= args.tile * args.tile_trigger:
        return written
    stride = int(args.tile * (1 - args.overlap))
    for oy in tile_origins(H, args.tile, stride):
        for ox in tile_origins(W, args.tile, stride):
            tw, th = min(args.tile, W - ox), min(args.tile, H - oy)
            lines = yolo_lines(sample.boxes, ox, oy, tw, th, tax, args.min_visible)
            if not lines and random.random() > args.keep_empty:
                continue
            write_image(img.crop((ox, oy, ox + tw, oy + th)), lines, out, split, f"{sample.uid}_t{ox}_{oy}")
            written += 1
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parent.parent
    ap.add_argument("--raw", type=Path, default=root / "data" / "raw")
    ap.add_argument("--out", type=Path, default=root / "data" / "yolo" / "detector")
    ap.add_argument("--tile", type=int, default=1280)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--tile-trigger", type=float, default=1.25, help="tile pages whose long side > tile * this")
    ap.add_argument("--min-visible", type=float, default=0.6, help="keep a clipped box if this fraction is inside the tile")
    ap.add_argument("--keep-empty", type=float, default=0.15, help="fraction of empty tiles to keep as background")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="only N samples per source (smoke test)")
    args = ap.parse_args()

    tax = Taxonomy.load()
    rng = random.Random(args.seed)
    random.seed(args.seed)
    unmapped: collections.Counter = collections.Counter()

    print("Loading annotations...")
    sources = {
        "pcbnet": (load_pcbnet(args.raw / "Dataset300", tax, unmapped), (0.8, 0.1, 0.1)),
        "ci2n": (load_ci2n(args.raw / "ci2n_datasets-main" / "device_identification", tax, unmapped), (0.85, 0.1, 0.05)),
    }
    if unmapped:
        print("WARNING unmapped labels (dropped):", dict(unmapped))

    if args.out.exists():
        shutil.rmtree(args.out)
    for split in ("train", "val", "test"):
        (args.out / "images" / split).mkdir(parents=True)
        (args.out / "labels" / split).mkdir(parents=True)

    splits_record: dict[str, dict[str, list[str]]] = {}
    class_counts = {s: collections.Counter() for s in ("train", "val", "test")}
    files = collections.Counter()
    with open(args.out / "annotations.jsonl", "w", encoding="utf-8") as ann:
        for name, (samples, ratios) in sources.items():
            if args.limit:
                samples = samples[: args.limit]
            parts = split_samples(samples, ratios, rng)
            splits_record[name] = {k: [s.uid for s in v] for k, v in parts.items()}
            for split, items in parts.items():
                for i, s in enumerate(items, 1):
                    files[split] += export(s, split, args.out, tax, args)
                    class_counts[split].update(b.cls for b in s.boxes)
                    w, h = Image.open(s.image).size
                    ann.write(json.dumps({
                        "uid": s.uid, "source": s.source, "split": split,
                        "image": str(s.image.relative_to(args.raw)), "width": w, "height": h,
                        "boxes": [b.__dict__ for b in s.boxes],
                    }) + "\n")
                    if i % 100 == 0:
                        print(f"  {name}/{split}: {i}/{len(items)}")

    (args.out / "splits.json").write_text(json.dumps(splits_record, indent=1), encoding="utf-8")
    (args.out / "dataset.yaml").write_text(
        yaml_dump(args.out, tax.classes), encoding="utf-8")

    print("\nImages written:", dict(files))
    print(f"{'class':<15}{'train':>8}{'val':>8}{'test':>8}")
    for c in tax.classes:
        print(f"{c:<15}{class_counts['train'][c]:>8}{class_counts['val'][c]:>8}{class_counts['test'][c]:>8}")
    print(f"\nDataset ready: {args.out / 'dataset.yaml'}")


def yaml_dump(out: Path, classes: list[str]) -> str:
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(classes))
    return f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names}\n"


if __name__ == "__main__":
    main()
