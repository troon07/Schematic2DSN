"""Score wire extraction + pin finding against PCBnet ground truth.

    python scripts/eval_wires.py --mode oracle   # true pin positions: scores wire tracing only
    python scripts/eval_wires.py --mode boxes    # only component boxes: also finds pins

Components come from ground truth here, so detector errors are excluded;
this isolates the wire/pin stage. Metrics, over every ground-truth pin that
touches a wire:
    pin_exact  share of pins whose set of connected pins is exactly right
    pair P/R   precision / recall over all "pin A connects to pin B" pairs
    edge_exact share of wire networks recovered exactly
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
from circuit2dsn.netlist import attach_points, find_pins  # noqa: E402
from circuit2dsn.pcbnet import load_design  # noqa: E402
from circuit2dsn.wires import build_graph, erase_boxes, erase_text, ink_mask  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parent.parent


def pairs(n: int) -> int:
    return n * (n - 1) // 2


def predict(design, rgb: np.ndarray, mode: str, shrink: float, text: bool = True, no_bars: bool = False,
            adaptive: bool = False, gray=None):
    """Return {gt_pin_id: predicted group key} plus the wire graph."""
    mask = ink_mask(rgb)
    if text:
        # Ground-truth text boxes stand in for the (future) text detector.
        mask = erase_text(mask, [t.box for t in design.texts.values()])
    if mode == "oracle":
        # Grow bodies slightly: the drawn outline of an IC sits just outside
        # the annotated body box and would otherwise join all of its pins.
        g = build_graph(erase_boxes(mask, [c.body for c in design.components.values()], shrink=-2.5))
        pts = {pid: (p.comp, *p.tip) for pid, p in design.pins.items()}
        pins = attach_points(g, pts, radius=6 + 2 * g.stroke)
        return {pid: p.group for pid, p in pins.items()}, g

    from circuit2dsn.tracing import trace

    g, found, _ = trace(mask, {cid: c.box for cid, c in design.components.items()},
                        {cid: c.cls for cid, c in design.components.items()},
                        steps=(shrink, 0.0) if adaptive else (shrink,), gray=gray,
                        bar_parts=None if no_bars else {cid for cid, c in design.components.items()
                                                        if c.cls in ("capacitor", "battery")})
    by_comp: dict[str, list] = {}
    for p in found.values():
        by_comp.setdefault(p.comp, []).append(p)
    tol = 6 + 2 * g.stroke
    out = {}
    for pid, gp in design.pins.items():
        best, bd = None, tol
        for p in by_comp.get(gp.comp, []):
            d = np.hypot(p.x - gp.tip[0], p.y - gp.tip[1])
            if d <= bd:
                best, bd = p, d
        out[pid] = best.group if best else None
    return out, g


def score(design, pred: dict[str, int | None]) -> dict:
    return score_groups({pid: p.edge for pid, p in design.pins.items() if p.edge is not None}, pred)


def score_groups(gt: dict[str, str], pred: dict) -> dict:
    """Compare two groupings of the same pins (gt: pin -> group, pred: pin -> group or None)."""
    key = {pid: (pred.get(pid) if pred.get(pid) is not None else f"none:{pid}") for pid in gt}
    gt_sets: dict = {}
    pr_sets: dict = {}
    for pid in gt:
        gt_sets.setdefault(gt[pid], set()).add(pid)
        pr_sets.setdefault(key[pid], set()).add(pid)
    cells = Counter((gt[p], key[p]) for p in gt)
    exact = sum(1 for p in gt if gt_sets[gt[p]] == pr_sets[key[p]])
    edge_ok = sum(1 for e, s in gt_sets.items() if pr_sets[key[next(iter(s))]] == s)
    return {
        "pins": len(gt), "pin_exact": exact,
        "tp": sum(pairs(n) for n in cells.values()),
        "gt_pairs": sum(pairs(len(s)) for s in gt_sets.values()),
        "pred_pairs": sum(pairs(len(s)) for s in pr_sets.values()),
        "edges": len(gt_sets), "edge_exact": edge_ok,
        "unattached": sum(1 for p in gt if pred.get(p) is None),
    }


def report(name: str, s: dict) -> str:
    p = s["tp"] / max(s["pred_pairs"], 1)
    r = s["tp"] / max(s["gt_pairs"], 1)
    return (f"{name:<34} pins={s['pins']:>5}  pin_exact={s['pin_exact'] / max(s['pins'], 1):6.1%}  "
            f"pairP={p:6.1%} pairR={r:6.1%}  edge_exact={s['edge_exact'] / max(s['edges'], 1):6.1%}  "
            f"unattached={s['unattached']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["oracle", "boxes"], default="oracle")
    ap.add_argument("--split", default="test", help="train / val / test / all")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shrink", type=float, default=2.0, help="px kept at box edge when erasing bodies")
    ap.add_argument("--no-text", action="store_true", help="do not erase text regions")
    ap.add_argument("--no-bars", action="store_true", help="old behaviour: plate remnants stay pins")
    ap.add_argument("--adaptive", action="store_true", help="re-trace parts that look wrong with the body fully erased")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "eval_wires")
    args = ap.parse_args()

    raw = ROOT / "data" / "raw" / "Dataset300"
    if args.split == "all":
        uids = sorted(p.name for p in raw.iterdir() if p.is_dir())
    else:
        splits = json.loads((ROOT / "data" / "yolo" / "detector" / "splits.json").read_text())
        uids = sorted(u.removeprefix("pcbnet_") for u in splits["pcbnet"][args.split])
    if args.limit:
        uids = uids[: args.limit]

    args.out.mkdir(parents=True, exist_ok=True)
    total = Counter()
    rows = []
    t0 = time.time()
    for uid in uids:
        design = load_design(raw / uid)
        rgb = np.asarray(Image.open(design.image).convert("RGB"))
        pred, _ = predict(design, rgb, args.mode, args.shrink, not args.no_text, args.no_bars, args.adaptive, rgb.min(2))
        s = score(design, pred)
        total.update(s)
        rows.append({"uid": uid, **s})
        print(report(f"{uid} {design.image.name[:25]}", s), flush=True)
    print("-" * 120)
    print(report(f"TOTAL {args.mode} ({len(uids)} designs, {time.time() - t0:.0f}s)", total))
    (args.out / f"{args.mode}_{args.split}.json").write_text(json.dumps({"total": total, "designs": rows}, indent=1))


if __name__ == "__main__":
    main()
