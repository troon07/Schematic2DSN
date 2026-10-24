"""Score full netlists (wires + ground + power names + net labels) on PCBnet.

    python scripts/eval_netlist.py              # test split
    python scripts/eval_netlist.py --no-labels  # wires + ground only

Parts and text come from ground truth (stand-ins for the detector and OCR),
so this measures the netlist builder. Metrics are as in eval_wires.py but
against electrical nets instead of drawn wire networks.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from circuit2dsn.pcbnet import gt_nets, load_design  # noqa: E402
from circuit2dsn.pipeline import Part, Text, extract_netlist  # noqa: E402
from circuit2dsn.taxonomy import Taxonomy  # noqa: E402
from circuit2dsn.textroles import role_of  # noqa: E402
from circuit2dsn.textstage import read_texts  # noqa: E402
from scripts.prepare_text_data import inside  # noqa: E402
from scripts.eval_wires import report, score_groups  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parent.parent


def symbol_name(design, comp) -> str | None:
    """Name of a power/port symbol from the annotated text link (stand-in for
    the text stage's text-to-component matching)."""
    if comp.cls not in ("power", "port"):
        return None
    for tid in comp.texts:
        t = design.texts[tid]
        if t.content.strip() and t.kind in ("netlabel", ""):
            return t.content.strip()
    return None


def design_texts(design, rgb, text_mode: str, reader=None) -> list[Text]:
    """Annotated text boxes + roles (stand-in for the text detector); content
    from annotations ("gt") or recognised by OCR ("ocr")."""
    link = {tid: c for c in design.components.values() for tid in c.texts}
    items = list(design.texts.values())
    roles = []
    for t in items:
        c = link.get(t.id)
        roles.append(role_of(t, c.cls if c else None, bool(c) and inside(t.box, c.body)))
    if text_mode == "ocr":
        return read_texts(rgb, [t.box for t in items], roles, reader)
    if text_mode == "none":
        return [Text("", t.box, "") for t in items]   # still erase text, but no names
    return [Text(t.content, t.box, r) for t, r in zip(items, roles)]


def run_design(design, text_mode: str = "gt", gt_match: bool = False, reader=None) -> tuple[dict, dict]:
    rgb = np.asarray(Image.open(design.image).convert("RGB"))
    parts = [Part(c.id, c.cls, c.box, name=symbol_name(design, c) if gt_match else None)
             for c in design.components.values() if c.cls]
    texts = design_texts(design, rgb, text_mode, reader)
    result = extract_netlist(rgb, parts, texts)

    # Map predicted pins to ground-truth pins: same part, nearest tip.
    tol = 6 + 2 * result["stroke"]
    by_part: dict[str, list] = {}
    for pid, p in result["pins"].items():
        by_part.setdefault(p["part"], []).append(p)
    pred = {}
    for pid, gp in design.pins.items():
        best, bd = None, tol
        for p in by_part.get(gp.comp, []):
            d = float(np.hypot(p["x"] - gp.tip[0], p["y"] - gp.tip[1]))
            if d <= bd:
                best, bd = p, d
        pred[pid] = best["net"] if best else None
    return pred, result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--text", choices=["gt", "ocr", "none"], default="gt",
                    help="text content: annotations, OCR on annotated boxes, or no text names")
    ap.add_argument("--gt-match", action="store_true", help="symbol names from annotated links, not textmatch")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "eval_netlist")
    args = ap.parse_args()

    tax = Taxonomy.load()
    raw = ROOT / "data" / "raw" / "Dataset300"
    if args.split == "all":
        uids = sorted(p.name for p in raw.iterdir() if p.is_dir())
    else:
        splits = json.loads((ROOT / "data" / "yolo" / "detector" / "splits.json").read_text())
        uids = sorted(u.removeprefix("pcbnet_") for u in splits["pcbnet"][args.split])
    if args.limit:
        uids = uids[: args.limit]

    args.out.mkdir(parents=True, exist_ok=True)
    total, rows, conf_stats = Counter(), [], Counter()
    reader = None
    if args.text == "ocr":
        from circuit2dsn.ocr import Reader
        reader = Reader()
    t0 = time.time()
    for uid in uids:
        design = load_design(raw / uid, tax)
        pred, result = run_design(design, args.text, args.gt_match, reader)
        gt = gt_nets(design)
        s = score_groups(gt, pred)
        total.update(s)
        rows.append({"uid": uid, **s})
        # Is low confidence where the errors are? (what the review UI relies on)
        gt_sets: dict = {}
        for p, n in gt.items():
            gt_sets.setdefault(n, set()).add(p)
        for net in result["nets"]:
            members = {p for p, n in pred.items() if n == net["id"] and p in gt}
            if not members:
                continue
            ok = all(gt_sets[gt[p]] == members for p in members)
            band = "high" if net["conf"] >= 0.85 else "low"
            conf_stats[(band, ok)] += 1
        print(report(f"{uid} {design.image.name[:25]}", s), flush=True)
    print("-" * 120)
    print(report(f"TOTAL nets text={args.text} ({len(uids)}, {time.time() - t0:.0f}s)", total))
    for band in ("high", "low"):
        ok, bad = conf_stats[(band, True)], conf_stats[(band, False)]
        print(f"  {band:>4}-confidence nets: {ok + bad:>5}  correct {ok / max(ok + bad, 1):6.1%}")
    name = f"netlist_{args.text}"
    (args.out / f"{name}_{args.split}.json").write_text(json.dumps({"total": total, "designs": rows}, indent=1))


if __name__ == "__main__":
    main()
