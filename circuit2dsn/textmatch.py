"""Attach texts to parts: designator, value, and the name of power/port symbols.

Rules from the PCBnet paper's matching stage:
- nearest text in the part's neighbourhood (box-to-box distance, scaled by
  part size so big ICs can have their label further away)
- category consistency: a designator's prefix and a value's unit must suit
  the part ("C12" / "100nF" do not go on a resistor)
- alignment bonus: labels usually sit straight beside or above the part
- uniqueness: global greedy assignment, best pairs first, one text per part
  and one part per text
Each match carries a confidence for the review UI.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

# Which designator prefixes suit which part class.
PREFIXES = {
    "resistor": ("R", "RN", "RP", "VR", "RV"),
    "capacitor": ("C", "EC", "TC", "CE"),
    "inductor": ("L", "FB"),
    "diode": ("D", "ZD", "TVS", "ESD"),
    "led": ("LED", "D", "L"),
    "bjt": ("Q", "T", "VT"),
    "mosfet": ("Q", "M", "T"),
    "opamp": ("U", "IC", "A"),
    "ic_box": ("U", "IC", "J", "P", "CN", "X", "H", "K", "SW", "Y", "M", "USB", "CON"),
    "switch": ("S", "SW", "K", "KEY"),
    "crystal": ("Y", "X", "XTAL", "OSC"),
    "fuse": ("F", "FU", "PTC"),
    "battery": ("BT", "B", "BAT"),
    "buzzer": ("BZ", "LS", "B", "BUZ"),
    "antenna": ("ANT", "E", "AE"),
    "transformer": ("T", "TR"),
    "terminal": ("TP", "J", "P", "H"),
    "jumper": ("JP", "J", "W"),
    "mounting_hole": ("H", "MH", "M"),
    "net_tie": ("NT", "R"),
    **{g: ("U", "IC", "G") for g in ("and_gate", "or_gate", "nand_gate", "nor_gate", "xor_gate", "xnor_gate",
                                     "not_gate", "buffer_gate", "optocoupler")},
    "potentiometer": ("RV", "VR", "R", "P"),
    "thyristor": ("Q", "T", "SCR", "D"),
    "triac": ("Q", "T", "TR"),
    "varistor": ("RV", "VDR", "R", "MOV"),
    "relay": ("K", "RL", "RLY"),
    "speaker": ("LS", "SP", "SPK"),
    "microphone": ("MK", "MIC"),
    "motor": ("M", "MOT"),
    "lamp": ("LA", "L", "H"),
    "meter": ("MES", "M", "V", "A"),
}
VALUE_UNIT = {"F": ("capacitor",), "H": ("inductor",), "Ω": ("resistor",), "R": ("resistor",)}
NO_VALUE = ("ground", "power", "port", "no_connect", "wire_hop", "mounting_hole")
NAMED = ("power", "port", "ground", "terminal")


@dataclass
class Match:
    part: str
    text: int          # index into the texts list
    kind: str          # "designator" / "value" / "name"
    conf: float


def _gap(a, b) -> float:
    dx = max(a[0] - b[2], b[0] - a[2], 0)
    dy = max(a[1] - b[3], b[1] - a[3], 0)
    return float(np.hypot(dx, dy))


def _aligned(a, b) -> bool:
    """Text straight beside (shares rows) or above/below (shares columns) the part."""
    rows = min(a[3], b[3]) - max(a[1], b[1]) > 0
    cols = min(a[2], b[2]) - max(a[0], b[0]) > 0
    return rows or cols


def prefix_ok(text: str, cls: str) -> bool | None:
    m = re.match(r"([A-Za-z]+)", text.strip())
    if not m or cls not in PREFIXES:
        return None
    return m.group(1).upper() in PREFIXES[cls]


def unit_ok(text: str, cls: str) -> bool | None:
    from .textfix import canonical_value

    v = canonical_value(text)
    m = re.fullmatch(r"[\d.]+[pnumkMG]?([FHΩR])\S*", v)
    if m:
        return cls in VALUE_UNIT.get(m.group(1), ())
    if re.fullmatch(r"[\d.]+[kM]", v):           # "10k", "4.7M": resistor shorthand
        return cls == "resistor"
    if re.fullmatch(r"[\d.]+[pnu]", v):          # "100n", "4.7u": capacitor shorthand
        return cls in ("capacitor", "inductor")
    return None


def _stacked(a, b) -> bool:
    """Two text boxes forming one label block: one right above the other
    (left-aligned or overlapping in x) or side by side on one row."""
    h = max(min(a[3] - a[1], b[3] - b[1]), 4)
    x_overlap = min(a[2], b[2]) - max(a[0], b[0]) > 0 or abs(a[0] - b[0]) < h
    y_gap = max(a[1] - b[3], b[1] - a[3], 0)
    y_overlap = min(a[3], b[3]) - max(a[1], b[1]) > 0
    x_gap = max(a[0] - b[2], b[0] - a[2], 0)
    return (x_overlap and y_gap < 1.2 * h) or (y_overlap and x_gap < 2 * h)


def _base_cost(p, t, kind):
    size = max(p.box[2] - p.box[0], p.box[3] - p.box[1])
    gap = _gap(p.box, t.box)
    if gap > 1.5 * size + 25:
        return None, None
    cost = gap / (size + 10)
    fit = prefix_ok(t.content, p.cls) if kind == "designator" else unit_ok(t.content, p.cls) if kind == "value" else True
    if fit is False:
        cost += 2.0
    elif fit is True:
        cost -= 0.3
    if not _aligned(p.box, t.box):
        cost += 0.3
    return cost, fit


def match_texts(parts, texts) -> list[Match]:
    """parts: objects with .id .cls .box; texts: objects with .content .box .role (.conf)."""
    # A part's designator and value are usually one label block ("R11" right
    # above "0.3 kOhm"). Find each value's likely part first, then prefer the
    # designator stacked with that value for the same part.
    value_owner = {}
    for ti, t in enumerate(texts):
        if t.role == "value" and t.content.strip():
            best = None
            for p in parts:
                if p.cls in NO_VALUE:
                    continue
                c, _ = _base_cost(p, t, "value")
                if c is not None and (best is None or c < best[0]):
                    best = (c, p.id)
            if best:
                value_owner[ti] = best[1]
    cands = []
    for ti, t in enumerate(texts):
        kind = {"designator": "designator", "value": "value", "net_label": "name"}.get(t.role)
        if kind is None or not t.content.strip():
            continue
        for p in parts:
            if kind == "name" and p.cls not in NAMED:
                continue
            if kind != "name" and p.cls in NO_VALUE:
                continue
            cost, fit = _base_cost(p, t, kind)
            if cost is None:
                continue
            if kind == "designator" and any(owner == p.id and _stacked(t.box, texts[vi].box)
                                            for vi, owner in value_owner.items()):
                cost -= 0.6
            cands.append((cost, p.id, ti, kind, fit))
    cands.sort(key=lambda c: c[0])
    used_text, used_slot, out = set(), set(), []
    for cost, pid, ti, kind, fit in cands:
        if ti in used_text or (pid, kind) in used_slot:
            continue
        used_text.add(ti)
        used_slot.add((pid, kind))
        conf = float(np.clip(1.0 - 0.35 * max(cost, 0), 0.3, 0.99)) * getattr(texts[ti], "conf", 1.0)
        if fit is False:
            conf = min(conf, 0.5)
        out.append(Match(pid, ti, kind, round(conf, 3)))
    return out


# ---------------------------------------------------------------- pin labels

@dataclass
class PinLabel:
    pin: str
    text: int
    kind: str          # "number" / "name"
    conf: float


def _side(box, x, y) -> str:
    d = {"L": abs(x - box[0]), "R": abs(x - box[2]), "T": abs(y - box[1]), "B": abs(y - box[3])}
    return min(d, key=d.get)


def match_pins(parts, pins: dict, texts) -> list[PinLabel]:
    """Pin numbers and pin names -> pins.

    parts: objects with .id .box; pins: {pid: obj with .comp .x .y};
    texts: objects with .content .box .role (pin_number / pin_name used).
    Geometry: a pin number sits beside the pin's stub, a pin name just inside
    the body; both line up with the pin along its side of the part.
    """
    box_of = {p.id: p.box for p in parts}
    wanted = {"pin_number": "number", "pin_name": "name"}
    raw = []      # (pid, ti, kind, signed along offset / thick, near / thick, (part, side))
    for ti, t in enumerate(texts):
        kind = wanted.get(t.role)
        if kind is None or not t.content.strip():
            continue
        x1, y1, x2, y2 = t.box
        thick = max(min(x2 - x1, y2 - y1), 4)
        for pid, q in pins.items():
            b = box_of.get(q.comp)
            if b is None:
                continue
            # text must be at/inside the part (slack of two text heights)
            if x2 < b[0] - 2 * thick or x1 > b[2] + 2 * thick or y2 < b[1] - 2 * thick or y1 > b[3] + 2 * thick:
                continue
            side = _side(b, q.x, q.y)
            if side in "LR":
                signed = (y1 + y2) / 2 - q.y
                near = (x1 - q.x) if side == "L" else (q.x - x2)        # depth of the text's nearest edge
                depth_span = b[2] - b[0]
            else:
                signed = (x1 + x2) / 2 - q.x
                near = (y1 - q.y) if side == "T" else (q.y - y2)
                depth_span = b[3] - b[1]
            if kind == "number":
                ok = -2 * thick <= near <= 0.35 * depth_span + thick
            else:
                ok = -thick <= near <= 0.6 * depth_span
            if ok and abs(signed) <= 1.8 * thick:
                raw.append((pid, ti, kind, signed / thick, near / thick, (q.comp, side)))

    # Numbers sit consistently on one side of their pin lines (above, below or
    # centred) within one side of a part; find that offset per part side so a
    # number between two close pins goes to the right one.
    # Measured on PCBnet train: names are centred (offset ~0.04 x text
    # height), numbers sit beside the line (median 0.64x).
    shift: dict = {}
    groups: dict = {}
    for pid, ti, kind, sg, nr, key in raw:
        if kind == "number":
            groups.setdefault(key, []).append((ti, sg))
    for key, items in groups.items():
        best = None
        for cand in (-0.65, 0.0, 0.65):
            per_text: dict = {}
            for ti, sg in items:
                per_text[ti] = min(per_text.get(ti, 9), abs(sg - cand))
            score = sum(1 for r in per_text.values() if r < 0.4) - 0.01 * sum(per_text.values())
            if best is None or score > best[0]:
                best = (score, cand)
        shift[key] = best[1]

    cands = []
    for pid, ti, kind, sg, nr, key in raw:
        if kind == "number":
            resid = abs(sg - shift.get(key, 0.0))
            if resid > 0.9:
                continue
            cost = resid + 0.5 * abs(nr) / 3
        else:
            if abs(sg) > 0.6:                    # names sit on the pin line
                continue
            cost = abs(sg) / 0.6 + 0.3 * max(nr, 0) / 4
        cands.append((cost, pid, ti, kind))
    cands.sort(key=lambda c: c[0])
    used_text, used_slot, out = set(), set(), []
    for cost, pid, ti, kind in cands:
        if ti in used_text or (pid, kind) in used_slot:
            continue
        used_text.add(ti)
        used_slot.add((pid, kind))
        conf = float(np.clip(1.0 - 0.4 * cost, 0.3, 0.99)) * getattr(texts[ti], "conf", 1.0)
        out.append(PinLabel(pid, ti, kind, round(conf, 3)))
    return out
