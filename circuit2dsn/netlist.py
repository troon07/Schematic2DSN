"""Pins + wire groups -> nets.

Two ways to get pins:
- `find_pins`: from component boxes only (what inference has). The box body
  is erased, so every wire that enters a component ends at the box edge; each
  such wire end is a pin.
- `attach_points`: from known pin positions (ground truth / library symbol
  pin maps), used to score the wire stage on its own.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .wires import WireGraph


@dataclass
class Pin:
    id: str
    comp: str
    x: float
    y: float
    group: int | None
    conf: float


@dataclass
class Net:
    id: str
    pins: list[str]
    label: str | None = None
    conf: float = 1.0


@dataclass
class Netlist:
    pins: dict[str, Pin] = field(default_factory=dict)
    nets: list[Net] = field(default_factory=list)

    def net_of(self) -> dict[str, str]:
        return {p: n.id for n in self.nets for p in n.pins}


def _bar_runs(graph: WireGraph, box, shrink: float) -> list[tuple]:
    """Bars of a symbol's own drawing left on the rim of its box after the body
    was erased: a capacitor plate, a diode's cathode bar, the top bar of a ground
    symbol ... A bar is a long straight run (>= 55% of that side) lying in the
    rim strip of one side, with its ends inside the box's extent.
    Returns (side, along_lo, along_hi, cross_lo, cross_hi, centre_x, centre_y)."""
    lab = graph.seg_labels
    H, W = lab.shape
    x1, y1, x2, y2 = box
    rim, ext = shrink + 2.5, 4
    bars = []
    for side in "LRTB":
        vertical = side in "LR"
        lo, hi = (y1, y2) if vertical else (x1, x2)
        n = hi - lo
        if n < 6:
            continue
        c0, c1 = {"L": (x1 - 0.5, x1 + rim), "R": (x2 - rim, x2 + 0.5),
                  "T": (y1 - 0.5, y1 + rim), "B": (y2 - rim, y2 + 0.5)}[side]
        a0, a1 = int(np.floor(lo)) - ext, int(np.ceil(hi)) + ext
        k0, k1 = int(np.floor(c0)), int(np.ceil(c1))
        along_max, cross_max = (H, W) if vertical else (W, H)
        a0c, a1c, k0c, k1c = max(a0, 0), min(a1, along_max - 1), max(k0, 0), min(k1, cross_max - 1)
        if a1c <= a0c or k1c < k0c:
            continue
        sub = lab[a0c:a1c + 1, k0c:k1c + 1] if vertical else lab[k0c:k1c + 1, a0c:a1c + 1].T
        occ = sub > 0
        prof = occ.any(1)
        best, i = None, 0
        while i < len(prof):
            if prof[i]:
                j = i
                while j + 1 < len(prof) and prof[j + 1]:
                    j += 1
                if best is None or j - i > best[1] - best[0]:
                    best = (i, j)
                i = j + 1
            else:
                i += 1
        if best is None:
            continue
        i, j = best
        if (j - i + 1) < max(5, 0.55 * n):
            continue
        if (i == 0 and a0c == a0) or (j == len(prof) - 1 and a1c == a1):
            continue                        # reaches past the box on a side: a wire passing by, not a bar
        cross = float(np.nonzero(occ[i:j + 1])[1].mean() + k0c)
        along = a0c + (i + j) / 2
        cx, cy = (cross, along) if vertical else (along, cross)
        bars.append((side, a0c + i, a0c + j, c0, c1, float(cx), float(cy)))
    return bars


def _bar_end(bars: list[tuple], px: float, py: float, tol: float) -> int | None:
    """Index of the bar whose end this wire end is, else None."""
    for bi, (side, a_lo, a_hi, c0, c1, _, _) in enumerate(bars):
        a, c = (py, px) if side in "LR" else (px, py)
        if c0 - 1 <= c <= c1 + 1 and (abs(a - a_lo) <= tol or abs(a - a_hi) <= tol):
            return bi
    return None


def find_pins(graph: WireGraph, boxes: dict[str, tuple[float, float, float, float]],
              shrink: float | dict[str, float], band: float | None = None, max_pins: dict[str, int] | None = None,
              stub_filter: bool = True, bar_parts: set[str] | None = None) -> dict[str, Pin]:
    """Wire ends lying on a component's box edge become that component's pins.

    - A wire end belongs to the single nearest box (dense layouts put another
      part's wire end within a few px of this box).
    - Ends of tiny stubs (leftover tips of an erased symbol's own lines) are
      not pins unless the stub joins two different parts that touch.
    - `max_pins` caps parts like ground / power symbols to their best pins:
      those on the longest wires.
    - `bar_parts`: ids of parts drawn with plates (capacitors, batteries). A
      plate left on the box rim after the body was erased has two tips that
      look like wire ends. A wire joins a plate in the middle, so the tips of
      one plate become ONE pin at its centre (unless that wire already has a
      real pin on the part). Only for these parts: on a resistor, clutter next
      to a tight box looks like a bar and would swallow a real lead.
    """
    band = band if band is not None else 3 + 2 * graph.stroke
    pts = graph.endpoints.astype(float)
    if len(pts) == 0 or not boxes:
        return {}
    merge = 2 * graph.stroke + 3
    ids = list(boxes)
    B = np.array([boxes[c] for c in ids], dtype=float)
    # `shrink` may differ per box (the px of each body kept at its edge when erased)
    S = np.array([shrink[c] if isinstance(shrink, dict) else shrink for c in ids], dtype=float)
    s_of = dict(zip(ids, S))
    shrink = float(S.mean())
    x, y = pts[:, :1], pts[:, 1:]
    outer = (x >= B[:, 0] - band) & (x <= B[:, 2] + band) & (y >= B[:, 1] - band) & (y <= B[:, 3] + band)
    inner = (x > B[:, 0] + S + 3) & (x < B[:, 2] - S - 3) & (y > B[:, 1] + S + 3) & (y < B[:, 3] - S - 3)
    # distance from each end to each box outline (0 on the edge)
    dx = np.maximum(np.maximum(B[:, 0] - x, x - B[:, 2]), 0)
    dy = np.maximum(np.maximum(B[:, 1] - y, y - B[:, 3]), 0)
    inside = (dx == 0) & (dy == 0)
    edge = np.where(inside, np.minimum.reduce([np.abs(x - B[:, 0]), np.abs(x - B[:, 2]),
                                               np.abs(y - B[:, 1]), np.abs(y - B[:, 3])]), np.hypot(dx, dy))
    ok = outer & ~inner
    edge = np.where(ok, edge, np.inf)

    # wire length per group (pixels), to spot tiny leftover stubs
    seg_len = np.bincount(graph.seg_labels.ravel(), minlength=len(graph.group_of_seg))
    group_len = np.bincount(graph.group_of_seg, weights=seg_len)
    min_len = 2 * band + shrink

    cands: list[tuple[str, float, float, int | None, float]] = []
    touch = S + 1.5        # an end this close to two boxes is a part-to-part contact
    for i in np.nonzero(ok.any(1))[0]:
        k = int(np.argmin(edge[i]))
        px, py = pts[i]
        g, _ = graph.group_at(px, py, 2.5)
        owners = [k] + [j for j in np.nonzero(edge[i] <= touch)[0] if j != k and edge[i, k] <= touch[k]]
        for j in owners:
            cands.append((ids[j], px, py, g, float(edge[i, j])))

    if bar_parts and cands:
        tol = merge + 1
        bars = {cid: (_bar_runs(graph, boxes[cid], s_of[cid]) if cid in bar_parts else [])
                for cid in {c[0] for c in cands}}
        kept, hits = [], defaultdict(list)
        for cand in cands:
            bi = _bar_end(bars[cand[0]], cand[1], cand[2], tol)
            if bi is None:
                kept.append(cand)
            else:
                hits[(cand[0], bi)].append(cand)
        real = {(c[0], c[3]) for c in kept}
        for (cid, bi), items in hits.items():
            g = items[0][3]
            if g is not None and (cid, g) in real:
                continue                    # this wire already reaches the part with a real pin
            _, _, _, _, _, cx, cy = bars[cid][bi]
            kept.append((cid, cx, cy, g, float(s_of[cid]) + 1.5))
            real.add((cid, g))
        cands = kept

    parts_on_group: dict[int, set] = defaultdict(set)
    for cid, _, _, g, _ in cands:
        if g is not None:
            parts_on_group[g].add(cid)

    taken: dict[str, list[tuple[float, float, int | None, float]]] = defaultdict(list)
    for cid, px, py, g, ed in cands:
        if stub_filter and g is not None and group_len[g] < min_len and len(parts_on_group[g]) < 2:
            continue
        if any(abs(px - tx) + abs(py - ty) < merge and g == tg for tx, ty, tg, _ in taken[cid]):
            continue
        # Far from the box edge -> less sure it is really this part's pin.
        sh = s_of[cid]
        conf = float(np.clip(1.0 - max(ed - sh, 0) / (band + sh + 1) * 0.5, 0.5, 0.99))
        taken[cid].append((px, py, g, conf))

    def off_centre(cid: str, px: float, py: float) -> float:
        # Leads of small symbols (ground stem, resistor ends) leave from the
        # middle of a box side; leftover tips of the symbol's lines sit at corners.
        x1, y1, x2, y2 = boxes[cid]
        mids = ((x1 + x2) / 2, y1), ((x1 + x2) / 2, y2), (x1, (y1 + y2) / 2), (x2, (y1 + y2) / 2)
        return min(np.hypot(px - mx, py - my) for mx, my in mids) / max(x2 - x1, y2 - y1, 1)

    pins: dict[str, Pin] = {}
    for cid, items in taken.items():
        cap = (max_pins or {}).get(cid)
        if cap is not None and len(items) > cap:
            # a pin on a wire that reaches other parts beats one on a loose scrap of the
            # symbol's own outline; among those, the lead leaving from a side's middle wins
            items = sorted(items, key=lambda t: (0 if t[2] is not None and len(parts_on_group.get(t[2], ())) >= 2 else 1,
                                                 off_centre(cid, t[0], t[1])))[:cap]
        for n, (px, py, g, conf) in enumerate(items, 1):
            pid = f"{cid}.{n}"
            pins[pid] = Pin(pid, cid, float(px), float(py), g, round(conf, 3))
    return pins


def attach_points(graph: WireGraph, points: dict[str, tuple[str, float, float]], radius: float) -> dict[str, Pin]:
    """Known pin positions -> pins with the wire group at each position."""
    pins = {}
    for pid, (cid, x, y) in points.items():
        g, d = graph.group_at(x, y, radius)
        conf = 0.0 if g is None else float(np.clip(1.0 - d / (radius + 1) * 0.5, 0.5, 0.99))
        pins[pid] = Pin(pid, cid, x, y, g, round(conf, 3))
    return pins


def build_nets(pins: dict[str, Pin]) -> Netlist:
    by_group: dict[int, list[str]] = defaultdict(list)
    nets = []
    for pid, p in pins.items():
        if p.group is None:
            nets.append(Net(f"N{len(nets) + 1}", [pid], conf=p.conf))
        else:
            by_group[p.group].append(pid)
    for members in by_group.values():
        conf = min(pins[m].conf for m in members)
        nets.append(Net(f"N{len(nets) + 1}", members, conf=round(conf, 3)))
    return Netlist(pins, nets)


def find_contacts(ink: np.ndarray, boxes: dict[str, tuple[float, float, float, float]],
                  max_overlap: float = 0.25, reach: float = 2.0) -> list[tuple[str, str, float, float]]:
    """Parts drawn touching each other (a GND symbol placed right on an IC pin):
    no wire is left between them once bodies are erased. Where two boxes touch
    or overlap a little and there is ink in the shared strip, that is a direct
    pin-to-pin contact. Heavily overlapping boxes are skipped (a detection
    overlap, not a contact)."""
    ids = list(boxes)
    B = np.array([boxes[i] for i in ids], dtype=float)
    if len(B) < 2:
        return []
    area = np.maximum(B[:, 2] - B[:, 0], 1) * np.maximum(B[:, 3] - B[:, 1], 1)
    h, w = ink.shape
    out = []
    order = np.argsort(B[:, 0])
    for ii, i in enumerate(order):
        for j in order[ii + 1:]:
            if B[j, 0] > B[i, 2] + reach:
                break
            x1, y1 = max(B[i, 0], B[j, 0]) - reach, max(B[i, 1], B[j, 1]) - reach
            x2, y2 = min(B[i, 2], B[j, 2]) + reach, min(B[i, 3], B[j, 3]) + reach
            if x2 <= x1 or y2 <= y1:
                continue
            ov = max(0, min(B[i, 2], B[j, 2]) - max(B[i, 0], B[j, 0])) * max(0, min(B[i, 3], B[j, 3]) - max(B[i, 1], B[j, 1]))
            if ov > max_overlap * min(area[i], area[j]):
                continue
            a, b = int(max(x1, 0)), int(max(y1, 0))
            c, d = int(min(np.ceil(x2), w)), int(min(np.ceil(y2), h))
            ys, xs = np.nonzero(ink[b:d, a:c])
            if len(xs) >= 2:
                out.append((ids[i], ids[j], float(xs.mean() + a), float(ys.mean() + b)))
    return out

