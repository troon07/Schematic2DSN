"""CGHD (hand-drawn circuits, PASCAL VOC) -> YOLO labels in the 47-class taxonomy.

    python scripts/prepare_cghd.py --src data/raw/cghd --out data/yolo/cghd

Images are resized so the long side is --max-side (photos are large; the
detector trains at ~1024-1280 anyway). Whole drafters are held out for
validation (--val-drafters) so the score reflects unseen handwriting.
Labels that are not parts (text, junction, crossover, ...) are dropped.
"""

from __future__ import annotations

import argparse
import collections
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from circuit2dsn.taxonomy import Taxonomy  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
IMG_EXT = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def find_image(drafter: Path, stem: str) -> Path | None:
    for ext in IMG_EXT:
        p = drafter / "images" / (stem + ext)
        if p.exists():
            return p
    return None


def convert(job):
    """One annotated image -> resized JPEG + YOLO label file. Returns (split, class counts)."""
    xml, img_path, split, name, out, max_side = job
    tax = Taxonomy.load()
    root = ET.parse(xml).getroot()
    boxes = []
    unmapped = collections.Counter()
    for obj in root.iter("object"):
        lbl = (obj.findtext("name") or "").strip()
        m = tax.map("cghd", lbl)
        if m is None:
            unmapped[lbl] += 1
            continue
        if m[0] is None:
            continue
        bb = obj.find("bndbox")
        x1, y1, x2, y2 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        boxes.append((tax.index(m[0]), min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
    # CGHD also marks tiny inversion bubbles on gate pins as "not";
    # those are not inverters and would teach a NOT inside every NAND.
    not_idx, small = tax.index("not_gate"), {tax.index("terminal"), tax.index("not_gate")}
    areas = sorted((b[3] - b[1]) * (b[4] - b[2]) for b in boxes if b[0] not in small)
    med = areas[len(areas) // 2] if areas else 0
    boxes = [b for b in boxes if not (b[0] == not_idx and (b[3] - b[1]) * (b[4] - b[2]) < 0.15 * med)]
    im = Image.open(img_path)
    W, H = im.size                       # boxes are in these original pixels
    im.draft("RGB", (max_side, max_side))   # JPEG: decode at reduced size (much faster)
    im = im.convert("RGB")
    s = min(1.0, max_side / max(im.size))
    if s < 1.0:
        im = im.resize((round(im.size[0] * s), round(im.size[1] * s)), Image.BILINEAR)
    im.save(out / "images" / split / f"{name}.jpg", quality=90)
    lines, counts = [], collections.Counter()
    for c, x1, y1, x2, y2 in boxes:
        x1, x2 = max(0, x1), min(W, x2)
        y1, y2 = max(0, y1), min(H, y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        lines.append(f"{c} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
        counts[tax.classes[c]] += 1
    (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return split, counts, unmapped


def main() -> None:
    import os
    import random
    from multiprocessing import Pool

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=Path("data/raw/cghd"))
    ap.add_argument("--out", type=Path, default=Path("data/yolo/cghd"))
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--val-drafters", default="28,29,30,31", help="comma-separated drafter numbers for val")
    ap.add_argument("--max-train", type=int, default=0, help="use a random sample of N training images (0 = all)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    args = ap.parse_args()

    val = {f"drafter_{v.strip()}" for v in args.val_drafters.split(",") if v.strip()}
    for split in ("train", "val"):
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)
    drafters = sorted(p for p in args.src.glob("drafter_*") if p.is_dir())
    if not drafters:
        sys.exit(f"no drafter_* folders under {args.src}")
    jobs = []
    for drafter in drafters:
        split = "val" if drafter.name in val else "train"
        for xml in sorted((drafter / "annotations").glob("*.xml")):
            img_path = find_image(drafter, xml.stem)
            if img_path is not None:
                name = f"cghd_{drafter.name.replace('-', 'm')}_{xml.stem}"
                jobs.append((xml, img_path, split, name, args.out, args.max_side))
    if args.max_train:
        train = [j for j in jobs if j[2] == "train"]
        random.Random(0).shuffle(train)
        jobs = train[: args.max_train] + [j for j in jobs if j[2] == "val"]
    print(f"converting {len(jobs)} images with {args.workers} workers...", flush=True)
    counts, unmapped, images = collections.Counter(), collections.Counter(), collections.Counter()
    with Pool(args.workers) as pool:
        for i, (split, c, u) in enumerate(pool.imap_unordered(convert, jobs, chunksize=8), 1):
            images[split] += 1
            counts.update(c)
            unmapped.update(u)
            if i % 250 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    print("images:", dict(images))
    if unmapped:
        print("WARNING unmapped CGHD labels (dropped):", dict(unmapped))
    print("parts per class:", dict(counts.most_common()))


if __name__ == "__main__":
    main()
