"""Score pin number / pin name matching on PCBnet.

    python scripts/eval_pinlabels.py            # annotated text content
    python scripts/eval_pinlabels.py --ocr      # OCR text content

Parts and text boxes/roles come from annotations (stand-ins for the
detectors), pins are found from part boxes as at inference. For every
annotated pin with a number (name): is the matched text the same?
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from circuit2dsn.pcbnet import load_design  # noqa: E402
from circuit2dsn.taxonomy import Taxonomy  # noqa: E402
from scripts.eval_netlist import run_design  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def norm(s: str) -> str:
    return "".join((s or "").split()).upper()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test")
    ap.add_argument("--ocr", action="store_true")
    args = ap.parse_args()
    tax = Taxonomy.load()
    splits = json.loads((ROOT / "data" / "yolo" / "detector" / "splits.json").read_text())
    uids = sorted(u.removeprefix("pcbnet_") for u in splits["pcbnet"][args.split])
    reader = None
    if args.ocr:
        from circuit2dsn.ocr import Reader
        reader = Reader()
    st = Counter()
    for uid in uids:
        d = load_design(ROOT / "data" / "raw" / "Dataset300" / uid, tax)
        _, nl = run_design(d, "ocr" if args.ocr else "gt", reader=reader)
        by_part = {}
        for pid, p in nl["pins"].items():
            by_part.setdefault(p["part"], []).append(p)
        tol = 6 + 2 * nl["stroke"]
        matched = set()
        for gp in d.pins.values():
            best, bd = None, tol
            for p in by_part.get(gp.comp, []):
                dist = float(np.hypot(p["x"] - gp.tip[0], p["y"] - gp.tip[1]))
                if dist <= bd:
                    best, bd = p, dist
            if best is None:
                for kind, tid in (("number", gp.number_text), ("name", gp.name_text)):
                    if tid:
                        st[(kind, "n")] += 1
                        st[(kind, "pin not found")] += 1
                continue
            matched.add(id(best))
            for kind, tid in (("number", gp.number_text), ("name", gp.name_text)):
                got = best.get(kind)
                if tid:
                    st[(kind, "n")] += 1
                    if got is None:
                        st[(kind, "no label")] += 1
                    elif norm(got[0]) == norm(d.texts[tid].content):
                        st[(kind, "ok")] += 1
                    else:
                        st[(kind, "wrong")] += 1
                elif got is not None:
                    st[(kind, "extra")] += 1
    for kind in ("number", "name"):
        n = st[(kind, "n")]
        print(f"pin {kind:<6} annotated={n:>5}  correct={st[(kind, 'ok')] / max(n, 1):6.1%}  "
              f"wrong={st[(kind, 'wrong')] / max(n, 1):6.1%}  no label={st[(kind, 'no label')] / max(n, 1):6.1%}  "
              f"pin not found={st[(kind, 'pin not found')] / max(n, 1):6.1%}  labels on unannotated pins={st[(kind, 'extra')]}")


if __name__ == "__main__":
    main()
