"""Page image + detected parts (+ recognised text) -> netlist with confidence.

Net building, in order:
1. wires: pins joined by drawn wire (wires.py / netlist.py)
2. ground: every ground symbol is the same net (GND)
3. power / port symbols: same name -> same net (name matched by textmatch.py)
4. net labels: wires carrying the same label text are the same net

Only text with role "net_label" merges wires. Values and designators
("100nF", "R1") also sit near wires; merging on those would short unrelated
nets. The text detector assigns roles (textroles.ROLES).

Designators and values are attached to parts too (for the EDIF output).

Every net carries `conf` (its weakest link) and `doubts` (why), so the
review UI can ask about exactly those links.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .netlist import Pin, find_contacts, find_pins
from .textmatch import match_pins, match_texts
from .wires import WireGraph, build_graph, erase_boxes, erase_text, ink_mask

GROUND = "ground"
# Single-pin symbols: extra wire ends near them are leftovers of the symbol's
# own lines or a neighbour's wire. (Capping 2-4 pin parts was tested and hurt:
# real parts vary, e.g. 3-pin pots, 3- or 4-pin MOSFETs.)
PIN_COUNT = {"ground": 1, "power": 1, "port": 1, "terminal": 1, "no_connect": 1}
NAMED_SYMBOLS = ("power", "port")
DOUBT_BELOW = 0.85
POWER_NAME = re.compile(r"^(GND|AGND|DGND|PGND|VSS|VCC|VDD|VEE|VBAT|[+-]?\d+(\.\d+)?V\d*)$", re.I)
MULTI_PIN = {"ic_box", "and_gate", "or_gate", "nand_gate", "nor_gate", "xor_gate", "xnor_gate", "not_gate",
             "buffer_gate", "opamp", "mosfet", "bjt"}
CONTACTS = True      # parts drawn touching each other connect directly
PLATE_CLASSES = {"capacitor", "battery"}     # drawn with plates: see netlist.find_pins(bar_parts=...)
NO_CONTACT = {"port", "net_tie", "power", "ground", "terminal", "no_connect"}


@dataclass
class Part:
    id: str
    cls: str
    box: tuple[float, float, float, float]
    conf: float = 1.0
    name: str | None = None   # symbol text (VCC, +3.3V) when the text stage matched it


@dataclass
class Text:
    content: str
    box: tuple[float, float, float, float]
    role: str = ""          # textroles.ROLES: designator, value, net_label, pin_number, ...
    conf: float = 1.0


@dataclass
class _Merge:
    reason: str
    conf: float


def norm_name(s: str) -> str:
    return re.sub(r"\s+", "", s).upper()


class _UF:
    def __init__(self):
        self.p: dict = {}

    def find(self, a):
        self.p.setdefault(a, a)
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def _group_near_box(g: WireGraph, box, radius: float) -> tuple[int | None, float]:
    """Wire group closest to a text box (labels sit on or just beside a wire)."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    probes = [(cx, cy), (cx, y2), (cx, y1), (x1, cy), (x2, cy), (x1, y2), (x2, y2), (x1, y1), (x2, y1)]
    best = (None, float("inf"))
    for px, py in probes:
        grp, d = g.group_at(px, py, radius)
        if grp is not None and d < best[1]:
            best = (grp, d)
    return best


from .tracing import trace  # noqa: E402
from .unknowns import find_unidentified  # noqa: E402


def wire_traces(g, groups: set, step: int = 4) -> dict[str, list]:
    """Drawn wire pixels per wire group, thinned to one point per step x step
    cell: lets the EDIF writer redraw wires along the original paths."""
    ys, xs = np.nonzero(g.seg_labels)
    if len(xs) == 0:
        return {}
    grp = g.group_of_seg[g.seg_labels[ys, xs]]
    keep = np.isin(grp, list(groups))
    xs, ys, grp = xs[keep] // step, ys[keep] // step, grp[keep]
    out: dict[str, list] = {}
    if len(xs):
        cells = np.unique(np.stack([grp, xs, ys], 1), axis=0)
        for gi, cx, cy in cells:
            out.setdefault(str(int(gi)), []).append([int(cx * step + step // 2), int(cy * step + step // 2)])
    return out


def extract_netlist(rgb: np.ndarray, parts: list[Part], texts: list[Text] = (), shrink: float = 2.0,
                    orienter=None, adaptive_shrink: bool = True) -> dict:
    """orienter: optional orientation.Orienter for pin roles of diodes / BJTs / MOSFETs.
    adaptive_shrink: parts whose pins look wrong (tracing.part_score) are traced
    again with their body erased completely (see tracing.py)."""
    mask = ink_mask(rgb)
    if texts:
        mask = erase_text(mask, [t.box for t in texts])
    ink = mask
    boxes = {p.id: p.box for p in parts}
    g, pins, _ = trace(mask, boxes, {p.id: p.cls for p in parts},
                       steps=(shrink, 0.0) if adaptive_shrink else (shrink,),
                       gray=rgb.min(2) if rgb.ndim == 3 else rgb,
                       max_pins={p.id: PIN_COUNT[p.cls] for p in parts if p.cls in PIN_COUNT},
                       bar_parts={p.id for p in parts if p.cls in PLATE_CLASSES})
    contact_pairs = []
    if CONTACTS:
        cls_of = {p.id: p.cls for p in parts}
        for k, (a, b, x, y) in enumerate(find_contacts(ink, boxes), 1):
            # Measured on PCBnet val+test: stacked single-pin symbols (ports,
            # net ties, power/ground) and resistor pairs touch without being
            # connected far more often than not.
            pair = {cls_of[a], cls_of[b]}
            if pair <= NO_CONTACT or pair == {"resistor"}:
                continue
            pa, pb = Pin(f"{a}.c{k}", a, x, y, None, 0.6), Pin(f"{b}.c{k}", b, x, y, None, 0.6)
            pins[pa.id], pins[pb.id] = pa, pb
            contact_pairs.append((pa, pb))
        # a single-pin symbol touching a part: the contact is its pin; drop
        # wire ends near it that lead nowhere else
        group_size: dict = {}
        for q in pins.values():
            if q.group is not None:
                group_size[q.group] = group_size.get(q.group, 0) + 1
        touched = {q.comp for pair in contact_pairs for q in pair}
        for pid in [pid for pid, q in pins.items()
                    if q.comp in touched and PIN_COUNT.get(cls_of[q.comp]) == 1 and ".c" not in pid
                    and (q.group is None or group_size.get(q.group, 0) <= 1)]:
            del pins[pid]
    part_of = {p.id: p for p in parts}

    uf = _UF()
    merges: dict = {}   # root-independent log: node -> list[_Merge]

    def node(pin) -> tuple:
        return ("g", pin.group) if pin.group is not None else ("p", pin.id)

    for pin in pins.values():
        uf.find(node(pin))

    def join(a, b, reason: str, conf: float):
        if uf.find(a) != uf.find(b):
            merges.setdefault(a, []).append(_Merge(reason, conf))
            merges.setdefault(b, []).append(_Merge(reason, conf))
        uf.union(a, b)

    for pa, pb in contact_pairs:
        join(node(pa), node(pb), f"{pa.comp} touches {pb.comp}", 0.6)

    # 2. ground symbols
    gnd = [node(p) for p in pins.values() if part_of[p.comp].cls == GROUND]
    for n in gnd[1:]:
        join(gnd[0], n, "ground symbols are one net", 0.97)

    # polarised parts: which pin is anode / cathode / base / ...
    orient: dict[str, dict] = {}
    if orienter is not None:
        from .orientation import DEVICE, pin_roles

        for p in parts:
            if p.cls not in DEVICE:
                continue
            side, mirror, oconf = orienter.classify(rgb, p.cls, p.box)
            if side is None:
                continue
            mine = {pid: (q.x, q.y) for pid, q in pins.items() if q.comp == p.id}
            orient[p.id] = {"side": side, "mirror": mirror, "conf": oconf,
                            "roles": pin_roles(p.cls, p.box, mine, side, mirror)}

    # pin numbers / names (for the EDIF pins)
    pin_labels: dict[str, dict] = {}
    for m in (match_pins(parts, pins, texts) if texts else []):
        pin_labels.setdefault(m.pin, {})[m.kind] = [texts[m.text].content.strip(), m.conf]

    # 3. power / port symbols: name given, else matched from text
    matches = match_texts(parts, texts) if texts else []
    attached: dict[str, dict] = {}
    used_texts = set()
    for m in matches:
        t = texts[m.text]
        attached.setdefault(m.part, {})[m.kind] = (t.content.strip(), m.conf)
        if m.kind == "name":
            used_texts.add(m.text)
    names: dict[str, str] = {}
    by_name: dict[str, list] = {}
    for p in parts:
        if p.cls not in NAMED_SYMBOLS:
            continue
        if p.name:
            name, conf = p.name, p.conf
        elif "name" in attached.get(p.id, {}):
            name, conf = attached[p.id]["name"]
        else:
            continue
        names[p.id] = name
        for pin in pins.values():
            if pin.comp == p.id:
                by_name.setdefault(norm_name(name), []).append((node(pin), conf))
    # 4. net labels on wires (texts already used as a symbol's name excluded)
    for ti, t in enumerate(texts):
        if t.role != "net_label" or ti in used_texts or not t.content.strip():
            continue
        h = min(t.box[2] - t.box[0], t.box[3] - t.box[1])
        grp, d = _group_near_box(g, t.box, radius=h + 4)
        if grp is None:
            continue
        uf.find(("g", grp))
        by_name.setdefault(norm_name(t.content), []).append((("g", grp), t.conf * (1 - d / (h + 5) * 0.3)))
    for name, members in by_name.items():
        for n, c in members[1:]:
            join(members[0][0], n, f"same name '{name}'", round(min(c, members[0][1]), 3))

    # Junction confidence per wire group (crossing / dot decisions).
    junction_doubts: dict[int, list[dict]] = {}
    for j in g.junctions:
        if j.conf >= DOUBT_BELOW:
            continue
        for grp in {int(g.group_of_seg[a]) for a in j.arms}:
            junction_doubts.setdefault(grp, []).append(
                {"what": f"{j.kind} at ({j.x:.0f},{j.y:.0f})", "conf": j.conf})

    nets: dict = {}
    for pin in pins.values():
        nets.setdefault(uf.find(node(pin)), []).append(pin.id)
    # Name each net after its most descriptive attached name (longest), so a
    # stray pin-number label like "P7" loses to "PLL_VDD".
    label_of: dict = {}
    for name, members in by_name.items():
        for n, _ in members:
            r = uf.find(n)
            if len(name) > len(label_of.get(r, "")):
                label_of[r] = name
    gnd_root = uf.find(gnd[0]) if gnd else None

    out_nets = []
    net_of_pin = {}
    cls_of = {p.id: p.cls for p in parts}
    for k, (root, members) in enumerate(sorted(nets.items(), key=lambda kv: -len(kv[1])), 1):
        doubts = []
        for pid in members:
            if pins[pid].conf < DOUBT_BELOW:
                doubts.append({"what": f"pin {pid} attachment", "conf": pins[pid].conf})
        groups = {pins[pid].group for pid in members if pins[pid].group is not None}
        for grp in groups:
            doubts.extend(junction_doubts.get(grp, []))
        for n, ms in merges.items():
            if uf.find(n) == root:
                doubts.extend({"what": m.reason, "conf": m.conf} for m in ms if m.conf < DOUBT_BELOW)
        name = "GND" if root == gnd_root else label_of.get(root) or f"N{k}"
        if not POWER_NAME.match(name):
            # Sanity rules for dense logic sheets: wires drawn as a bus or ones that touch merge into one
            # net. Two pins of one chip on one signal net, or a huge signal net, is rarely right: a person checks.
            per_part = Counter(pins[pid].comp for pid in members)
            for cid, cnt in per_part.items():
                if cnt >= 2 and cls_of.get(cid) in MULTI_PIN:
                    doubts.append({"what": f"{cnt} pins of {cid} share this net (check the bus lines)", "conf": 0.55})
            if len(members) > 12:
                doubts.append({"what": f"very large net ({len(members)} pins): probably several wires merged", "conf": 0.5})
        uniq = {d["what"]: d for d in doubts}
        doubts = sorted(uniq.values(), key=lambda d: d["conf"])
        conf = min([1.0] + [d["conf"] for d in doubts])
        nid = f"net{k}"
        for pid in members:
            net_of_pin[pid] = nid
        out_nets.append({"id": nid, "name": name, "pins": sorted(members), "conf": round(conf, 3), "doubts": doubts})

    return {
        "parts": [{"id": p.id, "cls": p.cls, "conf": p.conf, "box": list(p.box), "name": names.get(p.id),
                   "designator": attached.get(p.id, {}).get("designator"),
                   "orientation": {k: v for k, v in orient[p.id].items() if k != "roles"} if p.id in orient else None,
                   "value": attached.get(p.id, {}).get("value"),
                   "pins": sorted(pid for pid, q in pins.items() if q.comp == p.id)} for p in parts],
        "pins": {pid: {"part": q.comp, "x": round(q.x, 1), "y": round(q.y, 1), "net": net_of_pin[pid], "conf": q.conf,
                       "group": q.group,
                       "number": pin_labels.get(pid, {}).get("number"), "name": pin_labels.get(pid, {}).get("name"),
                       "role": ([orient[q.comp]["roles"][pid], orient[q.comp]["conf"]]
                                if q.comp in orient and pid in orient[q.comp]["roles"] else None)}
                 for pid, q in pins.items()},
        "nets": out_nets,
        "junctions": [{"x": round(j.x, 1), "y": round(j.y, 1), "kind": j.kind, "conf": j.conf} for j in g.junctions],
        "texts": [{"content": t.content, "box": [round(v, 1) for v in t.box], "role": t.role, "conf": t.conf}
                  for t in texts],
        "stroke": g.stroke,
        "unidentified": find_unidentified(g, [p.box for p in parts], [(q.x, q.y) for q in pins.values()],
                                          [t.box for t in texts], {q.group for q in pins.values() if q.group is not None}),
        "wires": wire_traces(g, {q.group for q in pins.values() if q.group is not None}),
        "width": int(rgb.shape[1]),
        "height": int(rgb.shape[0]),
    }
