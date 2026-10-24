"""One dataset.yaml over every prepared source (47 classes).

    python scripts/make_combined.py
    python scripts/make_combined.py --out data/yolo/super.yaml --sources detector cghd open_schematics

Sources (folders under data/yolo, each with images/{train,val,test}):
  detector         PCBnet + Image2Net (printed schematics)
  synthetic        KiCad-rendered pages, every class incl. gates and relays
  cghd             CGHD hand-drawn photos / scans
  digitize_hcd     Digitize-HCD hand-drawn scans
  open_schematics  real KiCad project sheets (Open Schematics)
Whichever exist are used. Test folders are listed too, so the held-out test
split of every source can be scored separately (scripts/eval_super.py).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from circuit2dsn.taxonomy import Taxonomy  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ("detector", "synthetic", "cghd", "digitize_hcd", "open_schematics")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, default=ROOT / "data" / "yolo")
    ap.add_argument("--out", type=Path, default=None, help="default: <base>/combined.yaml")
    ap.add_argument("--sources", nargs="*", default=list(SOURCES))
    args = ap.parse_args()
    names = Taxonomy.load().classes
    out = args.out or args.base / "combined.yaml"
    lists = {"train": [], "val": [], "test": []}
    used = []
    folders = []
    for src in args.sources:      # a source may be stored in chunks: open_schematics_000, _050, ...
        folders += sorted(d.name for d in args.base.glob(f"{src}*") if d.is_dir()
                          and (d.name == src or d.name[len(src):].startswith("_")))
    for src in folders:
        for split, lst in lists.items():
            d = args.base / src / "images" / split
            if d.exists() and any(d.iterdir()):
                lst.append(d.resolve().as_posix())
                if src not in used:
                    used.append(src)
    if not lists["train"]:
        sys.exit(f"no prepared datasets under {args.base}")
    lines = []
    for split, lst in lists.items():
        if lst:
            lines += [f"{split}:"] + [f"  - {p}" for p in lst]
    lines += ["names:"] + [f"  {i}: {n}" for i, n in enumerate(names)]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(names)} classes; sources {', '.join(used)}; train folders {len(lists['train'])} -> {out}")


if __name__ == "__main__":
    main()
