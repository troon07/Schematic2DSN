"""Netlist JSON (pipeline output) -> EDIF 2 0 0 schematic for OrCAD Capture.

Import in Capture: File > Import Design > EDIF tab > open this file, choose
where to save the .dsn, OK. (Structure follows the Dataset300 .eds files.)

Geometry
- Distance unit 1e-8 m: 254000 units = 100 mil = OrCAD's grid. Everything
  is snapped to that grid; off-grid pins break wiring in Capture.
- Pixel scale is chosen so the tightest typical pin pitch is >= 1 grid, so
  neighbouring pins never collapse onto one grid point.
- Each part becomes its own cell: a rectangle at the part's box, one pin per
  detected pin, on the box side nearest to where the wire met the part.

Connectivity, two styles:
- "labels" (default): every connected pin gets a 1-grid wire stub carrying
  the net's name. Capture joins same-named labels on a page, so the result
  does not depend on wire routing and cannot short nets by accident.
- "joined": one EDIF net per net listing all its pins (as the Dataset300
  .eds files do), with the same stubs and one name label.
Pins that cannot get a stub without touching another net are reported.
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

G = 254000            # 100 mil in EDIF distance units
TEXT_H = 254000

PREFIX = {"resistor": "R", "capacitor": "C", "inductor": "L", "diode": "D", "led": "LED", "bjt": "Q",
          "mosfet": "Q", "opamp": "U", "ic_box": "U", "ground": "GND", "power": "PWR", "vsource": "V",
          "isource": "I", "port": "P", "terminal": "TP", "net_tie": "NT", "no_connect": "NC", "switch": "SW",
          "crystal": "Y", "fuse": "F", "battery": "BT", "buzzer": "BZ", "antenna": "ANT", "transformer": "T",
          "jumper": "JP", "mounting_hole": "H", "other": "X",
          "and_gate": "U", "or_gate": "U", "nand_gate": "U", "nor_gate": "U", "xor_gate": "U", "xnor_gate": "U",
          "not_gate": "U", "buffer_gate": "U", "potentiometer": "RV", "thyristor": "Q", "triac": "Q",
          "varistor": "RV", "optocoupler": "U", "relay": "K", "speaker": "LS", "microphone": "MK", "motor": "M",
          "lamp": "LA", "meter": "MES"}
SKIP_CLASSES = {"wire_hop"}   # a crossing marker, not a part


# ---------------------------------------------------------------- s-expressions

class Str(str):
    """An EDIF string literal (quoted on output)."""


def ascii_text(s: str) -> str:
    s = s.replace("Ω", "").replace("μ", "u").replace("µ", "u").replace("±", "+/-").replace("°", "deg")
    return "".join(c if 32 <= ord(c) < 127 else "?" for c in s).strip()


def quote(s: str) -> str:
    # EDIF escapes characters as %<ascii code>%
    s = ascii_text(s).replace("%", "%37%").replace('"', "%34%")
    return f'"{s}"'


def emit(node, indent: int = 0, out: list | None = None) -> list[str]:
    out = [] if out is None else out
    pad = "  " * indent
    if not isinstance(node, (list, tuple)):
        out.append(pad + (quote(node) if isinstance(node, Str) else str(node)))
        return out
    head, *rest = node
    flat = all(not isinstance(r, (list, tuple)) for r in rest)
    small = flat or (len(rest) <= 3 and all(not isinstance(r, (list, tuple)) or
                                            all(not isinstance(x, (list, tuple)) for x in r) for r in rest))
    if small:
        out.append(pad + _inline(node))
        return out
    first = [r for r in rest if not isinstance(r, (list, tuple))]
    out.append(pad + "(" + " ".join([str(head)] + [quote(r) if isinstance(r, Str) else str(r) for r in first]))
    for r in rest:
        if isinstance(r, (list, tuple)):
            emit(r, indent + 1, out)
    out.append(pad + ")")
    return out


def _inline(node) -> str:
    if not isinstance(node, (list, tuple)):
        return quote(node) if isinstance(node, Str) else str(node)
    return "(" + " ".join(_inline(n) for n in node) + ")"


def pt(x: int, y: int):
    return ("pt", int(x), int(y))


class Names:
    """Legal, unique EDIF identifiers: [A-Za-z][A-Za-z0-9_]*."""

    def __init__(self):
        self.used: set[str] = set()

    def make(self, prefix: str, raw: str) -> str:
        base = re.sub(r"[^A-Za-z0-9_]", "_", f"{prefix}_{raw}" if prefix else raw)[:200] or "N"
        if not base[0].isalpha():
            base = "N" + base
        name, k = base, 1
        while name.upper() in self.used:
            k += 1
            name = f"{base}_{k}"
        self.used.add(name.upper())
        return name


# ---------------------------------------------------------------- layout

@dataclass
class PartLayout:
    part: dict
    box: list[int]                                   # grid: x1, y1, x2, y2 (y down)
    pins: dict[str, tuple[str, int]] = field(default_factory=dict)   # pid -> (side, pos)
    ident: str = ""
    cell: str = ""
    refdes: str = ""

    def connect(self, pid: str) -> tuple[int, int]:
        """Grid point where a wire attaches to the pin (1 grid outside the body)."""
        side, pos = self.pins[pid]
        x1, y1, x2, y2 = self.box
        return {"L": (x1 - 1, pos), "R": (x2 + 1, pos), "T": (pos, y1 - 1), "B": (pos, y2 + 1)}[side]

    def stub_end(self, pid: str) -> tuple[int, int]:
        side, _ = self.pins[pid]
        x, y = self.connect(pid)
        dx, dy = {"L": (-1, 0), "R": (1, 0), "T": (0, -1), "B": (0, 1)}[side]
        return x + dx, y + dy


def choose_scale(netlist: dict) -> float:
    """EDIF units per pixel: tightest typical pin pitch -> at least one grid."""
    by_part = defaultdict(list)
    for p in netlist["pins"].values():
        by_part[p["part"]].append((p["x"], p["y"]))
    pitches = []
    for pts in by_part.values():
        if len(pts) >= 2:
            a = np.array(pts)
            d = np.hypot(*(a[:, None, :] - a[None, :, :]).transpose(2, 0, 1))
            d[d == 0] = np.inf
            pitches.append(d.min())
    pitch = float(np.percentile(pitches, 10)) if pitches else 30.0
    upp = max(25400.0, G / max(pitch, 1.0) * 1.05)
    # keep the page under ~100 inches on its long side
    longest = max(netlist.get("width", 2000), netlist.get("height", 2000))
    return min(upp, 254_000_000 / longest)


def layout_parts(netlist: dict, upp: float) -> list[PartLayout]:
    pins_of = defaultdict(list)
    for pid, p in netlist["pins"].items():
        pins_of[p["part"]].append(pid)
    out = []
    for part in netlist["parts"]:
        if part["cls"] in SKIP_CLASSES:
            continue
        x1, y1, x2, y2 = (v * upp / G for v in part["box"])
        box = [int(np.floor(x1)), int(np.floor(y1)), int(np.ceil(x2)), int(np.ceil(y2))]
        box[2] = max(box[2], box[0] + 1)
        box[3] = max(box[3], box[1] + 1)
        lay = PartLayout(part, box)
        taken: dict[str, set] = defaultdict(set)
        for pid in sorted(pins_of[part["id"]]):
            p = netlist["pins"][pid]
            gx, gy = p["x"] * upp / G, p["y"] * upp / G
            d = {"L": abs(gx - box[0]), "R": abs(gx - box[2]), "T": abs(gy - box[1]), "B": abs(gy - box[3])}
            side = min(d, key=d.get)
            along = gy if side in "LR" else gx
            pos = _free_slot(lay, side, int(round(along)), taken[side])
            taken[side].add(pos)
            lay.pins[pid] = (side, pos)
        out.append(lay)
    return out


def _free_slot(lay: PartLayout, side: str, want: int, taken: set) -> int:
    """Nearest unused grid position on a box side, growing the box if full."""
    while True:
        lo, hi = (lay.box[1], lay.box[3]) if side in "LR" else (lay.box[0], lay.box[2])
        want = min(max(want, lo), hi)
        for step in range(0, hi - lo + 1):
            for cand in (want - step, want + step):
                if lo <= cand <= hi and cand not in taken:
                    return cand
        if side in "LR":
            lay.box[3] += 1
        else:
            lay.box[2] += 1


def pin_order(pid: str):
    head, _, tail = pid.rpartition(".")
    return (head, int(tail)) if tail.isdigit() else (pid, 0)


def pin_numbers(netlist: dict, pids: list[str]) -> dict[str, str]:
    """Detected pin number when confident and unique on the part; else the
    next free number (OrCAD needs unique pin numbers per part)."""
    from .orientation import STANDARD_NUMBER

    out, used = {}, set()
    for pid in pids:
        num = netlist["pins"][pid].get("number")
        text = ascii_text(num[0]).replace(" ", "") if num and num[1] >= 0.5 else ""
        if text and text.upper() not in used:
            out[pid] = text
            used.add(text.upper())
    # polarised parts without readable numbers: standard numbers from pin roles
    if not used:
        for pid in pids:
            role = netlist["pins"][pid].get("role")
            std = STANDARD_NUMBER.get(role[0]) if role and role[1] >= 0.5 else None
            if std and std not in used:
                out[pid] = std
                used.add(std)
    k = 1
    for pid in pids:
        if pid not in out:
            while str(k).upper() in used:
                k += 1
            out[pid] = str(k)
            used.add(str(k))
    return out


# ---------------------------------------------------------------- writer

@dataclass
class EdifReport:
    parts: int = 0
    nets: int = 0
    stubs: int = 0
    unrouted: int = 0                                  # schematic style: connections drawn as stub + alias
    pins_without_stub: list = field(default_factory=list)
    shorted_pins: list = field(default_factory=list)   # pins of different nets on one grid point
    scale_units_per_px: float = 0.0
    pin_map: dict = field(default_factory=dict)        # pin id -> (instance identifier, pin number)
    refdes: dict = field(default_factory=dict)         # instance identifier -> reference designator


def display(text: str, x: int, y: int, group: str = "ATTRIBUTE", justify: str = "UPPERLEFT"):
    return ("stringDisplay", Str(text),
            ("display",
             ("figureGroupOverride", group, ("textHeight", TEXT_H), ("userData", "visibleName", ("false",)),
              ("visible", ("true",))),
             ("justify", justify), ("orientation", "R0"), ("origin", pt(x, y))))


def emitted_nets(netlist: dict, lays: list[PartLayout]):
    """(net, [(layout, pin id)]) for every net that gets written: >= 2 pins or a real name."""
    lay_of = {l.part["id"]: l for l in lays}
    for net in netlist["nets"]:
        members = [(lay_of[netlist["pins"][pid]["part"]], pid) for pid in net["pins"]
                   if netlist["pins"][pid]["part"] in lay_of and pid in lay_of[netlist["pins"][pid]["part"]].pins]
        named = bool(net.get("name")) and not re.fullmatch(r"N\d+", net.get("name") or "")
        if len(members) >= 2 or (named and members):
            yield net, members


def plan_stubs(netlist: dict, lays: list[PartLayout]):
    """Stub segment per pin, or None where it would touch another net.
    Returns (stub_of, shorted_pins, pins_without_stub)."""
    lay_of = {l.part["id"]: l for l in lays}
    occupied: dict[tuple[int, int], str] = {}
    shorted, nostub, stub_of = [], [], {}
    # connection points first: pins of different nets on one grid point = short
    for net in netlist["nets"]:
        for pid in net["pins"]:
            l = lay_of.get(netlist["pins"][pid]["part"])
            if l is None or pid not in l.pins:
                continue
            c = l.connect(pid)
            if c in occupied and occupied[c] != net["id"]:
                shorted.append(pid)
            occupied.setdefault(c, net["id"])
    for net, members in emitted_nets(netlist, lays):
        for l, pid in members:
            a, b = l.connect(pid), l.stub_end(pid)
            if occupied.get(b, net["id"]) != net["id"]:
                nostub.append(pid)
                stub_of[pid] = None
                continue
            occupied[b] = net["id"]
            stub_of[pid] = (a, b)
    return stub_of, shorted, nostub


def split_pages(netlist: dict) -> list[tuple[int, dict]]:
    """One sub-netlist per page (multipage.merge_pages output has "pages").
    A net keeps its id and name on every page it reaches."""
    pages = netlist.get("pages") or []
    if len(pages) <= 1:
        return [(1, netlist)]
    out = []
    for pg in pages:
        n = pg["page"]
        parts = [p for p in netlist["parts"] if p.get("page", 1) == n]
        ids = {p["id"] for p in parts}
        pins = {pid: q for pid, q in netlist["pins"].items() if q["part"] in ids}
        nets = []
        for net in netlist["nets"]:
            here = [q for q in net["pins"] if q in pins]
            if here:
                nets.append({**net, "pins": here})
        out.append((n, {"parts": parts, "pins": pins, "nets": nets,
                        "width": pg.get("width", 0), "height": pg.get("height", 0)}))
    return out


def _layout_page(sub: dict, rep: EdifReport):
    """Grid layout for one page: spread out until no two nets share a grid
    point, then slide any pins still colliding (parts drawn touching)."""
    base, best = choose_scale(sub), None
    for factor in (1, 1.5, 2, 3, 4, 6):
        lays = layout_parts(sub, base * factor)
        stub_of, shorted, nostub = plan_stubs(sub, lays)
        score = 10 * len(shorted) + len(nostub)
        if best is None or score < best[0]:
            best = (score, base * factor, lays, stub_of, shorted, nostub)
        if score == 0:
            break
    _, upp, lays, stub_of, shorted, nostub = best
    for _ in range(5):
        if not shorted and not nostub:
            break
        lay_of = {l.part["id"]: l for l in lays}
        for pid in shorted + nostub:
            l = lay_of.get(sub["pins"][pid]["part"])
            if l is None or pid not in l.pins:
                continue
            side, pos = l.pins[pid]
            taken = {q for k, (sd, q) in l.pins.items() if sd == side and k != pid}
            l.pins[pid] = (side, _free_slot(l, side, pos + 1, taken | {pos}))
        stub_of, shorted, nostub = plan_stubs(sub, lays)
    rep.shorted_pins += shorted
    rep.pins_without_stub += nostub
    rep.scale_units_per_px = max(rep.scale_units_per_px, round(upp, 1))
    return upp, lays, stub_of


def write_edif(netlist: dict, design_name: str = "circuit", style: str = "labels") -> tuple[str, EdifReport]:
    """Multi-page netlists (with "pages") become one EDIF page per page; nets
    reaching several pages carry the same name on each."""
    if style not in ("labels", "joined"):
        raise ValueError("style must be 'labels' or 'joined'")
    rep = EdifReport()
    names = Names()
    lib, top = "CIRCUIT2DSN_LIB", "SCHEMATIC_TOP"
    names.make("", lib), names.make("", top)
    pages = [(n, sub, *_layout_page(sub, rep)) for n, sub in split_pages(netlist)]

    # reference designators: matched text if confident and unique in the design
    seen = defaultdict(int)
    for _, _, _, lays, _ in pages:
        for l in lays:
            d = l.part.get("designator")
            if d and d[1] >= 0.5:
                seen[ascii_text(d[0]).upper()] += 1
    for _, _, _, lays, _ in pages:
        for l in lays:
            d = l.part.get("designator")
            text = ascii_text(d[0]) if d and d[1] >= 0.5 else ""
            l.refdes = text if text and seen[text.upper()] == 1 else PREFIX.get(l.part["cls"], "X") + "?"
            l.ident = names.make("INST", l.part["id"])
            l.cell = names.make("CELL", l.part["id"])

    # net names, unique in the design (a net spanning pages keeps one name)
    net_names, used_names = {}, set()
    for net in netlist["nets"]:
        raw = ascii_text(net.get("name") or "") or net["id"]
        name, k = raw, 1
        while name.upper() in used_names:
            k += 1
            name = f"{raw}_{k}"
        used_names.add(name.upper())
        net_names[net["id"]] = name

    cells, page_nodes, emitted = [], [], set()
    for page_no, sub, upp, lays, stub_of in pages:
        all_x = [v for l in lays for v in (l.box[0] - 3, l.box[2] + 3)] + [sub.get("width", 0) * upp / G]
        all_y = [v for l in lays for v in (l.box[1] - 3, l.box[3] + 3)] + [sub.get("height", 0) * upp / G]
        W, H = int(np.ceil(max(all_x + [10]))) + 2, int(np.ceil(max(all_y + [10]))) + 2

        def ex(gx):    # grid -> EDIF x
            return int(gx) * G

        def ey(gy, H=H):    # grid (y down) -> EDIF y (y up)
            return (H - int(gy)) * G

        port_ids: dict[tuple[str, str], str] = {}
        for l in lays:
            bw, bh = l.box[2] - l.box[0], l.box[3] - l.box[1]
            ports, impls = [], []
            numbers = pin_numbers(sub, sorted(l.pins, key=pin_order))
            for n, pid in enumerate(sorted(l.pins, key=pin_order), 1):
                port = f"P{n}"
                number = numbers[pid]
                nm = sub["pins"][pid].get("name") or sub["pins"][pid].get("role")
                pin_name = ascii_text(nm[0]) if nm and nm[1] >= 0.5 and ascii_text(nm[0]) else number
                port_ids[(l.part["id"], pid)] = port
                rep.pin_map[pid] = (l.ident, number)
                ports.append(("port", ("rename", port, Str(pin_name)), ("direction", "INOUT"), ("designator", Str(number))))
                side, pos = l.pins[pid]
                cx, cy = l.connect(pid)
                # symbol coordinates: origin at body top-left, y negative downward
                sx, sy = (cx - l.box[0]) * G, -(cy - l.box[1]) * G
                bx, by = {"L": (0, sy), "R": (bw * G, sy), "T": (sx, 0), "B": (sx, -bh * G)}[side]
                impls.append(("portImplementation", ("name", port),
                              ("connectLocation", ("figure", "GRAPHICS", ("dot", pt(sx, sy))))))
                impls.append(("figure", "PIN", ("path", ("pointList", pt(bx, by), pt(sx, sy)))))
            body = ("figure", "BODY_FGP", ("rectangle", pt(0, 0), pt(bw * G, -bh * G)))
            label = ("keywordDisplay", "designator",
                     ("display", ("figureGroupOverride", "ATTRIBUTE", ("textHeight", TEXT_H)),
                      ("justify", "LOWERLEFT"), ("orientation", "R0"), ("origin", pt(0, G // 2))))
            cells.append(("cell", l.cell, ("cellType", "GENERIC"),
                          ("view", "VIEW_1", ("viewType", "SCHEMATIC"),
                           ("interface", *ports,
                            ("designator", Str(PREFIX.get(l.part["cls"], "X") + "?")),
                            ("property", "VALUE", ("string", Str(""))),
                            ("symbol", ("boundingBox", ("rectangle", pt(0, 0), pt(bw * G, -bh * G))),
                             body, label, *impls)))))

        instances = []
        for l in lays:
            value = l.part.get("value")
            vtext = ascii_text(value[0]) if value and value[1] >= 0.5 else ""
            ox, oy = ex(l.box[0]), ey(l.box[1])
            instances.append(("instance", ("rename", l.ident, Str(l.refdes)),
                              ("viewRef", "VIEW_1", ("cellRef", l.cell, ("libraryRef", lib))),
                              ("transform", ("orientation", "R0"), ("origin", pt(ox, oy))),
                              ("designator", display(l.refdes, ox, oy + G // 2, justify="LOWERLEFT")),
                              ("property", "VALUE", ("string", display(vtext, ex(l.box[2]) + G // 2, oy))),
                              ("property", "CLASS", ("string", Str(l.part["cls"]))),
                              ("property", "DETECT_CONF", ("string", Str(f"{l.part.get('conf', 1.0):.2f}"))),
                              ("property", "SOURCE_ID", ("string", Str(l.part["id"])))))
            rep.parts += 1

        net_nodes = []
        for net, members in emitted_nets(sub, lays):
            name = net_names[net["id"]]
            stubs = [(l, pid, stub_of.get(pid)) for l, pid in members]
            emitted.add(net["id"])
            if style == "joined":
                ident = names.make("NET", f"{net['id']}_p{page_no}")
                first = next((st for st in stubs if st[2]), None)
                lx, ly = (first[2][1] if first else (0, 0))
                node = ["net", ("rename", ident, display(name, ex(lx), ey(ly), "LABEL", "LOWERLEFT")),
                        ("joined", *[("portRef", port_ids[(l.part["id"], pid)], ("instanceRef", l.ident))
                                     for l, pid, _ in stubs])]
                for l, pid, seg in stubs:
                    if seg:
                        (ax, ay), (bx, by) = seg
                        node.append(("figure", "NET", ("path", ("pointList", pt(ex(ax), ey(ay)), pt(ex(bx), ey(by))))))
                        rep.stubs += 1
                net_nodes.append(tuple(node))
            else:
                for i, (l, pid, seg) in enumerate(stubs, 1):
                    ident = names.make("NET", f"{net['id']}_{i}")
                    ref = ("joined", ("portRef", port_ids[(l.part["id"], pid)], ("instanceRef", l.ident)))
                    if seg:
                        (ax, ay), (bx, by) = seg
                        node = ("net", ("rename", ident, display(name, ex(bx), ey(by), "LABEL", "LOWERLEFT")), ref,
                                ("figure", "NET", ("path", ("pointList", pt(ex(ax), ey(ay)), pt(ex(bx), ey(by))))))
                        rep.stubs += 1
                    else:
                        node = ("net", ("rename", ident, Str(name)), ref)
                    net_nodes.append(node)
        page_nodes.append(("page", f"PAGE_{page_no}", ("pageSize", ("rectangle", pt(0, 0), pt(W * G, H * G))),
                           *instances, *net_nodes))
    rep.nets = len(emitted)

    t = time.localtime()
    tree = ("edif", names.make("", design_name),
            ("edifVersion", 2, 0, 0), ("edifLevel", 0), ("keywordMap", ("keywordLevel", 0)),
            ("status", ("written", ("timestamp", t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec),
                        ("author", Str("circuit2dsn")), ("program", Str("circuit2dsn edif.py"), ("version", Str("1.0"))))),
            ("library", lib, ("edifLevel", 0),
             ("technology",
              ("numberDefinition", ("scale", 1, ("e", 1, -8), ("unit", "DISTANCE")), ("gridMap", 1, 1)),
              ("figureGroup", "BODY_FGP", ("color", 50, 0, 50)),
              ("figureGroup", "ATTRIBUTE", ("color", 9, 25, 63), ("textHeight", TEXT_H)),
              ("figureGroup", "PIN", ("color", 37, 37, 37)),
              ("figureGroup", "NET", ("color", 12, 34, 86)),
              ("figureGroup", "LABEL", ("color", 9, 25, 63), ("textHeight", TEXT_H)),
              ("figureGroup", "GRAPHICS")),
             *cells,
             ("cell", top, ("cellType", "GENERIC"),
              ("view", "SCHEMATIC_VIEW", ("viewType", "SCHEMATIC"), ("interface",),
               ("contents", *page_nodes)))),
            ("design", ("rename", "DESIGN", Str(design_name)), ("cellRef", top, ("libraryRef", lib))))
    return "\n".join(emit(tree)) + "\n", rep
