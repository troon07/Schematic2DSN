"""Score text-to-part matching on PCBnet (annotated text boxes, roles and
content stand in for the text detector and OCR; this measures matching only).

    python scripts/eval_textmatch.py

For every part that has an annotated designator / value / symbol name, is
the matched text the annotated one?
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from circuit2dsn.pcbnet import load_design  # noqa: E402
from circuit2dsn.taxonomy import Taxonomy  # noqa: E402
from circuit2dsn.textmatch import NAMED, match_texts  # noqa: E402
from circuit2dsn.textroles import role_of  # noqa: E402
from scripts.prepare_text_data import inside  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class P:
    id: str
    cls: str
    box: tuple


@dataclass
class T:
    id: str
    content: str
    box: tuple
    role: str
    conf: float = 1.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    tax = Taxonomy.load()
    splits = json.loads((ROOT / "data" / "yolo" / "detector" / "splits.json").read_text())
    uids = sorted(u.removeprefix("pcbnet_") for u in splits["pcbnet"][args.split])
    stats = Counter()
    for uid in uids:
        d = load_design(ROOT / "data" / "raw" / "Dataset300" / uid, tax)
        link = {tid: c for c in d.components.values() for tid in c.texts}
        texts = []
        for t in d.texts.values():
            c = link.get(t.id)
            texts.append(T(t.id, t.content, t.box, role_of(t, c.cls if c else None, bool(c) and inside(t.box, c.body))))
        parts = [P(c.id, c.cls, c.box) for c in d.components.values() if c.cls]
        got = {(m.part, m.kind): texts[m.text].id for m in match_texts(parts, texts)}
        for c in d.components.values():
            if not c.cls:
                continue
            for tid in c.texts:
                role = next(t.role for t in texts if t.id == tid)
                kind = {"designator": "designator", "value": "value"}.get(role)
                if role == "net_label" and c.cls in NAMED:
                    kind = "name"
                if kind is None:
                    continue
                stats[(kind, "n")] += 1
                stats[(kind, "ok")] += got.get((c.id, kind)) == tid
                stats[(kind, "missing")] += (c.id, kind) not in got
    for kind in ("designator", "value", "name"):
        n = stats[(kind, "n")]
        print(f"{kind:<11} parts={n:>5}  correct={stats[(kind, 'ok')] / max(n, 1):6.1%}  "
              f"no match={stats[(kind, 'missing')] / max(n, 1):6.1%}")


if __name__ == "__main__":
    main()
