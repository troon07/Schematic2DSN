"""Build the YOLO text-detector dataset from PCBnet: one box per text region,
class = its role (designator, value, net_label, pin_number, pin_name,
part_name, other).

Uses the same per-design train/val/test split as the component detector
(data/yolo/detector/splits.json), so test designs stay unseen by both.
Text is small (median ~14 px tall), so large pages are tiled at native
resolution exactly like the component data.

    python scripts/prepare_text_data.py
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from circuit2dsn.pcbnet import load_design  # noqa: E402
from circuit2dsn.taxonomy import Taxonomy  # noqa: E402
from circuit2dsn.textroles import ROLES, role_of  # noqa: E402
from scripts.prepare_detector_data import Box, Sample, export, yaml_dump  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


class RoleTaxonomy:
    """Minimal stand-in for Taxonomy so the shared exporter can index roles."""
    classes = ROLES

    def index(self, name: str) -> int:
        return ROLES.index(name)


def inside(box, outer, tol: float = 2) -> bool:
    return box[0] >= outer[0] - tol and box[1] >= outer[1] - tol and box[2] <= outer[2] + tol and box[3] <= outer[3] + tol


def load_samples(raw: Path, tax: Taxonomy) -> dict[str, Sample]:
    out = {}
    for folder in sorted(p for p in raw.iterdir() if p.is_dir()):
        d = load_design(folder, tax)
        link = {tid: c for c in d.components.values() for tid in c.texts}
        s = Sample(f"pcbnet_{folder.name}", "pcbnet", d.image)
        for t in d.texts.values():
            if not t.content.strip() or t.box[2] <= t.box[0] or t.box[3] <= t.box[1]:
                continue
            c = link.get(t.id)
            role = role_of(t, c.cls if c else None, bool(c) and inside(t.box, c.body))
            s.boxes.append(Box(role, t.kind or None, *t.box, t.content))
        out[s.uid] = s
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "raw" / "Dataset300")
    ap.add_argument("--splits", type=Path, default=ROOT / "data" / "yolo" / "detector" / "splits.json")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "yolo" / "text")
    ap.add_argument("--tile", type=int, default=1280)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--tile-trigger", type=float, default=1.25)
    ap.add_argument("--min-visible", type=float, default=0.9, help="text cut by a tile edge is unreadable; keep only near-whole boxes")
    ap.add_argument("--keep-empty", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    samples = load_samples(args.raw, Taxonomy.load())
    splits = json.loads(args.splits.read_text())["pcbnet"]

    if args.out.exists():
        shutil.rmtree(args.out)
    for split in ("train", "val", "test"):
        (args.out / "images" / split).mkdir(parents=True)
        (args.out / "labels" / split).mkdir(parents=True)

    tax = RoleTaxonomy()
    counts = {s: collections.Counter() for s in ("train", "val", "test")}
    files = collections.Counter()
    with open(args.out / "annotations.jsonl", "w", encoding="utf-8") as ann:
        for split, uids in splits.items():
            for i, uid in enumerate(uids, 1):
                s = samples[uid]
                files[split] += export(s, split, args.out, tax, args)
                counts[split].update(b.cls for b in s.boxes)
                ann.write(json.dumps({"uid": uid, "split": split, "image": str(s.image.relative_to(args.raw.parent)),
                                      "boxes": [b.__dict__ for b in s.boxes]}, ensure_ascii=False) + "\n")
                if i % 50 == 0:
                    print(f"  {split}: {i}/{len(uids)}", flush=True)

    (args.out / "dataset.yaml").write_text(yaml_dump(args.out, ROLES), encoding="utf-8")
    print("\nImages written:", dict(files))
    print(f"{'role':<12}{'train':>8}{'val':>8}{'test':>8}")
    for r in ROLES:
        print(f"{r:<12}{counts['train'][r]:>8}{counts['val'][r]:>8}{counts['test'][r]:>8}")
    print(f"\nDataset ready: {args.out / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
