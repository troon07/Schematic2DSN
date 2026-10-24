"""Consistency check of the pin finder on Image2Net (ci2n) images.

    python scripts/eval_pins_proxy.py --n 600 --variant old
    python scripts/eval_pins_proxy.py --n 600 --variant rim

These images have device boxes but no connection ground truth, so this checks
things that must hold for any correct result:
  count_ok   the part has the number of pins its symbol has (2 for a resistor ...)
  dup        two pins of one part sit on the SAME wire: for a 2-terminal part
             this means the part is shorted (the "capacitor with two pins on
             one net" defect: a leftover plate / arc of the symbol read as wires)
  <2 pins    fewer than two pins attached for a part that needs two
Lower dup and higher count_ok = better. Nothing here needs ground truth, so it
can run on all 2,269 images; use it to compare two versions of find_pins.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from circuit2dsn.netlist import find_pins  # noqa: E402
from circuit2dsn.pipeline import PIN_COUNT, PLATE_CLASSES  # noqa: E402
from circuit2dsn.taxonomy import Taxonomy  # noqa: E402
from circuit2dsn.wires import build_graph, erase_boxes, erase_text, ink_mask  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "data" / "raw" / "ci2n_datasets-main" / "device_identification"
EXPECT = {"resistor": (2,), "capacitor": (2,), "inductor": (2,), "diode": (2,), "vsource": (2,), "isource": (2,),
          "mosfet": (3, 4), "bjt": (3,), "ground": (1,), "power": (1,), "port": (1,), "switch": (2, 3),
          "opamp": (2, 3, 4), "battery": (2,)}
SHRINK = 2.0
CACHE = ROOT / "outputs" / "fix_cap" / "ocr_cache"


def text_boxes(stem: str, rgb: np.ndarray) -> list:
    """Text found by RapidOCR, as the pipeline does (cached: OCR is the slow part)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{stem}.json"
    if f.exists():
        return json.loads(f.read_text())
    from circuit2dsn.textstage import ocr_page

    boxes = [list(t.box) for t in ocr_page(rgb)]
    f.write_text(json.dumps(boxes))
    return boxes


def page_mask(jf, rgb: np.ndarray, erase_texts: bool = True) -> np.ndarray:
    mask = ink_mask(rgb)
    return erase_text(mask, text_boxes(Path(jf).stem, rgb)) if erase_texts else mask


def one(args):
    jf, variant, steps = args
    tax = Taxonomy.load()
    d = json.loads(Path(jf).read_text())
    img = DIR / d["imagePath"]
    if not img.exists():
        img = DIR / (Path(jf).stem + ".png")
    if not img.exists():
        img = DIR / (Path(jf).stem + ".jpg")
    rgb = np.asarray(Image.open(img).convert("RGB"))
    boxes, cls = {}, {}
    for k, s in enumerate(d["shapes"]):
        m = tax.map("ci2n", s["label"])
        if not m or m[0] is None or s["shape_type"] != "rectangle":
            continue
        (x1, y1), (x2, y2) = s["points"][0], s["points"][1]
        cid = f"C{k}"
        boxes[cid], cls[cid] = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)), m[0]
    if not boxes:
        return Counter()
    from circuit2dsn.tracing import trace

    g, pins, _ = trace(page_mask(jf, rgb), boxes, cls, steps=steps, gray=rgb.min(2),
                       max_pins={c: PIN_COUNT[k] for c, k in cls.items() if k in PIN_COUNT},
                       bar_parts={c for c, k in cls.items() if k in PLATE_CLASSES} if variant == "bars" else None)
    by = defaultdict(list)
    for p in pins.values():
        by[p.comp].append(p)
    out = Counter()
    parts_on = defaultdict(set)
    for p in pins.values():
        if p.group is not None:
            parts_on[p.group].add(p.comp)
    for cid, k in cls.items():
        if k == "wire_hop" or k not in EXPECT:
            continue
        ps = by.get(cid, [])
        if EXPECT[k] == (1,):          # one-pin symbol: its pin must sit on a wire that reaches another part
            out[(k, "lone")] += not any(p.group is not None and len(parts_on[p.group]) >= 2 for p in ps)
        groups = [p.group for p in ps if p.group is not None]
        out[(k, "parts")] += 1
        out[(k, "count_ok")] += len(ps) in EXPECT[k]
        out[(k, "dup")] += len(groups) != len(set(groups)) and EXPECT[k][0] >= 2
        out[(k, "lt2")] += len(set(groups)) < 2 and EXPECT[k][0] >= 2
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--variant", default="old", help="old | bars | rim")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", default="2", help="erase margins to try, e.g. 2 or 2,0 (adaptive)")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    files = sorted(DIR.glob("*.json"))
    random.Random(args.seed).shuffle(files)
    files = files[: args.n]
    tot = Counter()
    with Pool(args.workers) as pool:
        for c in pool.imap_unordered(one, [(str(f), args.variant, tuple(float(v) for v in args.steps.split(','))) for f in files], chunksize=4):
            tot.update(c)
    print(f"variant={args.variant}  steps={args.steps}  images={len(files)}")
    print(f"{'class':<10}{'parts':>7}{'count_ok':>10}{'dup':>8}{'<2 pins':>9}{'pin alone':>11}")
    allp = alld = allc = 0
    for k in EXPECT:
        n = tot[(k, "parts")]
        if not n:
            continue
        lone = f"{tot[(k, 'lone')] / n:>11.1%}" if EXPECT[k] == (1,) else ""
        print(f"{k:<10}{n:>7}{tot[(k, 'count_ok')] / n:>10.1%}{tot[(k, 'dup')] / n:>8.1%}{tot[(k, 'lt2')] / n:>9.1%}{lone}")
        if EXPECT[k][0] >= 2:
            allp += n
            alld += tot[(k, "dup")]
            allc += tot[(k, "count_ok")]
    print(f"2+ terminal parts: {allp}  count_ok {allc / max(allp, 1):.1%}  dup {alld / max(allp, 1):.1%}")


if __name__ == "__main__":
    main()
