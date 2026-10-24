"""OCR accuracy on PCBnet text regions (annotated boxes, so detection is excluded).

    python scripts/eval_ocr.py --limit-texts 1500

Reports exact-match accuracy and character error rate (CER), overall and per
text role, after whitespace is ignored. Writes the worst misreads to
outputs/eval_ocr/ for building correction rules.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402

from circuit2dsn.ocr import Reader, wire_lines  # noqa: E402
from circuit2dsn.pcbnet import load_design  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parent.parent


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def norm(s: str) -> str:
    return "".join(s.split())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit-texts", type=int, default=1500, help="random sample of text regions")
    ap.add_argument("--fix", action="store_true", help="apply circuit2dsn.textfix corrections")
    ap.add_argument("--no-lines", dest="lines", action="store_false", help="do not whiten wire lines in crops")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "eval_ocr")
    args = ap.parse_args()

    splits = json.loads((ROOT / "data" / "yolo" / "detector" / "splits.json").read_text())
    uids = sorted(u.removeprefix("pcbnet_") for u in splits["pcbnet"][args.split])
    items = []
    for uid in uids:
        d = load_design(ROOT / "data" / "raw" / "Dataset300" / uid)
        items += [(d.image, t) for t in d.texts.values() if norm(t.content)]
    random.Random(0).shuffle(items)
    items = items[: args.limit_texts]
    items.sort(key=lambda it: str(it[0]))

    fix = None
    if args.fix:
        from circuit2dsn.textfix import clean, same
        fix = clean

    reader = Reader()
    stats: dict[str, Counter] = defaultdict(Counter)
    misses = []
    page, page_path, lines = None, None, None
    t0 = time.time()
    for image, t in items:
        if image != page_path:
            page, page_path = np.asarray(Image.open(image).convert("RGB")), image
            lines = wire_lines(page, text_height=14) if args.lines else None
        r = reader.read(page, t.box, lines)
        pred = fix(r.text, t.kind, r.conf) if fix else r.text
        gt, pr = norm(t.content), norm(pred)
        e = edit_distance(gt, pr)
        hit = same(t.content, pred, t.kind) if fix else gt == pr
        for key in ("ALL", t.kind or "(none)"):
            s = stats[key]
            s["n"] += 1
            s["exact"] += hit
            s["chars"] += len(gt)
            s["errs"] += e
        if not hit:
            misses.append({"gt": t.content, "pred": pred, "raw": r.text, "conf": r.conf, "kind": t.kind,
                           "image": str(image.name), "box": t.box})
    print(f"{len(items)} texts in {time.time() - t0:.0f}s")
    for key, s in sorted(stats.items(), key=lambda kv: -kv[1]["n"]):
        print(f"{key:<12} n={s['n']:>5}  exact={s['exact'] / s['n']:6.1%}  CER={s['errs'] / max(s['chars'], 1):6.1%}")
    args.out.mkdir(parents=True, exist_ok=True)
    name = "misses_fixed.json" if args.fix else "misses.json"
    (args.out / name).write_text(json.dumps(misses, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
