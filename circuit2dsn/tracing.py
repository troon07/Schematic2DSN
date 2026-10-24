"""Trace wires and find pins with a per-part erase margin chosen by the result.

Why: each part's body is erased before the wires are traced. How much to erase
depends on how tight the detector's box is, and that differs between drawing
styles. With a loose box (PCBnet style) the symbol sits inside it and keeping
2 px at the edge is right. With a tight box (Image2Net style) the symbol's own
strokes (zig-zag tips, capacitor plates, circle arcs, a diode bar) lie ON the
edge and survive as fake wire ends: a capacitor ends up with 2 pins on one net.
No single margin fits both (measured: 2 px is best on PCBnet, 0 px on Image2Net).

So: trace with the safe margin, check every part against what its symbol must
look like (a resistor has 2 pins, on 2 different wires; a MOSFET 3-4 ...), and
re-trace only the parts that fail with their body erased completely. A part
keeps the tighter erase only if it came out better than before.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from .netlist import Pin, find_pins
from .wires import WireGraph, build_graph, erase_boxes

# number of pins a symbol of this class has (any of the listed counts is fine)
EXPECTED = {"resistor": (2,), "capacitor": (2,), "inductor": (2,), "diode": (2,), "led": (2,), "vsource": (2,),
            "isource": (2,), "battery": (2,), "fuse": (2,), "crystal": (2,), "buzzer": (2,), "switch": (2, 3),
            "mosfet": (3, 4), "bjt": (3,), "varistor": (2,), "lamp": (2,), "speaker": (2,), "microphone": (2,),
            "motor": (2,), "meter": (2,), "potentiometer": (3,), "thyristor": (3,), "triac": (3,),
            # one-pin symbols: a port whose tail joins the wire at its corner has no pin until re-traced
            "port": (1,), "terminal": (1,), "ground": (1,), "power": (1,)}
TWO_TERMINAL = {k for k, v in EXPECTED.items() if v == (2,)}
ONE_PIN = {"port", "terminal", "ground", "power"}


def part_score(cls: str, pins: list[Pin]) -> float:
    """0 = looks like its symbol. Higher = worse (pin count off, or a 2-terminal
    part with both pins on one wire)."""
    exp = EXPECTED.get(cls)
    if exp is None:
        return 0.0
    n = len(pins)
    score = float(min(abs(n - e) for e in exp))
    if cls in TWO_TERMINAL:
        groups = [p.group for p in pins if p.group is not None]
        score += float(len(groups) - len(set(groups)))
    return score


def trace(mask: np.ndarray, boxes: dict[str, tuple], classes: dict[str, str], steps: tuple[float, ...] = (2.0, 0.0),
          gray: np.ndarray | None = None, max_pins: dict[str, int] | None = None,
          bar_parts: set[str] | None = None) -> tuple[WireGraph, dict[str, Pin], dict[str, float]]:
    """mask: ink with text already erased. Returns (graph, pins, shrink used per part)."""
    ids = list(boxes)
    shrink = {c: steps[0] for c in ids}

    def run(sh: dict[str, float]):
        m = mask
        for s in sorted(set(sh.values())):
            subset = [boxes[c] for c in ids if sh[c] == s]
            if subset:
                m = erase_boxes(m, subset, s)
        g = build_graph(m, gray=gray)
        return g, find_pins(g, boxes, sh, max_pins=max_pins, bar_parts=bar_parts)

    def scores(pins: dict[str, Pin]) -> dict[str, float]:
        by: dict[str, list[Pin]] = {c: [] for c in ids}
        for p in pins.values():
            by[p.comp].append(p)
        return {c: part_score(classes.get(c, ""), by[c]) for c in ids}

    def done(g, pins, sh):
        repair_one_pin_parts(g, boxes, classes, pins)
        return g, pins, sh

    g, pins = run(shrink)
    if len(steps) < 2:
        return done(g, pins, shrink)
    s0 = scores(pins)
    bad = [c for c in ids if s0[c] > 0]
    if not bad:
        return done(g, pins, shrink)
    trial = dict(shrink)
    for c in bad:
        trial[c] = steps[1]
    g2, pins2 = run(trial)
    s1 = scores(pins2)
    for c in bad:                                  # keep the tighter erase only where it helped
        if s1[c] >= s0[c]:
            trial[c] = steps[0]
    if all(trial[c] == steps[1] for c in bad):
        return done(g2, pins2, trial)
    if all(trial[c] == steps[0] for c in bad):
        return done(g, pins, shrink)
    g3, pins3 = run(trial)
    return done(g3, pins3, trial)


def repair_one_pin_parts(graph: WireGraph, boxes: dict[str, tuple], classes: dict[str, str],
                         pins: dict[str, Pin], reach: float = 2.0) -> int:
    """A port / ground / power symbol exists to connect to a wire. If its pin sits
    on a wire that reaches no other part (a scrap of the symbol's own outline, or
    no pin at all because its short tail merged into the wire), use the connected
    wire that touches the symbol's box instead: the pin is the touching point
    closest to the box centre. Returns how many parts were repaired."""
    parts_on: dict[int, set] = {}
    for p in pins.values():
        if p.group is not None:
            parts_on.setdefault(p.group, set()).add(p.comp)
    by: dict[str, list[Pin]] = {}
    for p in pins.values():
        by.setdefault(p.comp, []).append(p)
    lab, h, w = graph.seg_labels, *graph.seg_labels.shape
    fixed = 0
    for cid, cls in classes.items():
        if cls not in ONE_PIN or cid not in boxes:
            continue
        mine = by.get(cid, [])
        if any(p.group is not None and len(parts_on.get(p.group, ())) >= 2 for p in mine):
            continue                                   # already on a connected wire
        x1, y1, x2, y2 = boxes[cid]
        a, b, c, d = max(int(np.floor(x1 - reach)), 0), max(int(np.floor(y1 - reach)), 0),             min(int(np.ceil(x2 + reach)) + 1, w), min(int(np.ceil(y2 + reach)) + 1, h)
        ys, xs = np.nonzero(lab[b:d, a:c])
        if not len(xs):
            continue
        xs, ys = xs + a, ys + b
        groups = graph.group_of_seg[lab[ys, xs]]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        best = None
        for g_id in set(int(v) for v in groups):
            if len(parts_on.get(g_id, set()) - {cid}) < 1:
                continue                                # reaches no other part
            sel = groups == g_id
            dist = np.hypot(xs[sel] - cx, ys[sel] - cy)
            k = int(np.argmin(dist))
            if best is None or dist[k] < best[0]:
                best = (float(dist[k]), g_id, float(xs[sel][k]), float(ys[sel][k]))
        if best is None:
            continue
        for p in mine:
            pins.pop(p.id, None)
        pid = f"{cid}.1"
        pins[pid] = Pin(pid, cid, best[2], best[3], best[1], 0.7)
        fixed += 1
    return fixed


def summary(shrink: dict[str, float], default: float) -> Counter:
    return Counter("tight" if v != default else "default" for v in shrink.values())
