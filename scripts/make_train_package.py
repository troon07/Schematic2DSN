"""Assemble train_package/: code + every prepared dataset + launcher, then zip it.

    python scripts/make_train_package.py            # copy into train_package/
    python scripts/make_train_package.py --zip      # also write train_package.zip

Checks before copying: every dataset folder has images and matching labels,
no label is out of the page, and the class ids fit the 47 classes.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from circuit2dsn.taxonomy import Taxonomy  # noqa: E402
from scripts.make_combined import SOURCES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "train_package"
IMG = {".jpg", ".jpeg", ".png", ".bmp"}


def check(folder: Path, n_classes: int) -> dict:
    out = {}
    for split in ("train", "val", "test"):
        imgs = [p for p in (folder / "images" / split).glob("*") if p.suffix.lower() in IMG] \
            if (folder / "images" / split).exists() else []
        bad = missing = boxes = 0
        for im in imgs:
            lab = folder / "labels" / split / (im.stem + ".txt")
            if not lab.exists():
                missing += 1
                continue
            for line in lab.read_text().split("\n"):
                if not line.strip():
                    continue
                c, x, y, w, h = line.split()
                c, x, y, w, h = int(c), float(x), float(y), float(w), float(h)
                boxes += 1
                if not (0 <= c < n_classes) or x - w / 2 < -1e-4 or y - h / 2 < -1e-4 or x + w / 2 > 1 + 1e-4 \
                        or y + h / 2 > 1 + 1e-4 or w <= 0 or h <= 0:
                    bad += 1
        if imgs:
            out[split] = {"images": len(imgs), "boxes": boxes, "no_label": missing, "bad_boxes": bad}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", action="store_true")
    args = ap.parse_args()
    n = len(Taxonomy.load().classes)
    base = ROOT / "data" / "yolo"
    folders = []
    for src in SOURCES:
        folders += sorted(d for d in base.glob(f"{src}*") if d.is_dir() and (d.name == src or d.name[len(src):].startswith("_")))
    total_img = 0
    problems = []
    for d in folders:
        rep = check(d, n)
        imgs = sum(v["images"] for v in rep.values())
        total_img += imgs
        print(f"{d.name:<22} " + "  ".join(f"{k}: {v['images']} img / {v['boxes']} boxes"
                                          + (f" ({v['no_label']} no label, {v['bad_boxes']} bad)" if v['no_label'] or v['bad_boxes'] else "")
                                          for k, v in rep.items()))
        if any(v["bad_boxes"] or v["no_label"] for v in rep.values()):
            problems.append(d.name)
    print(f"total {total_img} images in {len(folders)} folders; problems: {problems or 'none'}")

    for sub in ("circuit2dsn", "scripts", "configs"):
        shutil.copytree(ROOT / sub, PKG / sub, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "review_ui"))
    for d in folders:
        dst = PKG / "data" / "yolo" / d.name
        shutil.copytree(d, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.cache", "debug_*", ".shards", "stats.json"))
    if (ROOT / "requirements.txt").exists():
        shutil.copy(ROOT / "requirements.txt", PKG)
    print(f"package ready: {PKG}")
    ck = PKG / "start_checkpoint" / "super" / "weights" / "last.pt"
    print("start checkpoint:", "present" if ck.exists() else "MISSING (see start_checkpoint/PUT_COLAB_FILES_HERE.txt)")
    if args.zip:
        z = ROOT / "train_package.zip"
        with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:   # images are compressed already
            for f in PKG.rglob("*"):
                if f.is_file() and ".venv" not in f.parts and "work" not in f.parts and "RESULTS" not in f.parts:
                    zf.write(f, Path("train_package") / f.relative_to(PKG))
        print(f"zip: {z} ({z.stat().st_size / 1e9:.1f} GB)")


if __name__ == "__main__":
    main()
