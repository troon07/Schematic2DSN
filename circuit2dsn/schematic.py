"""Netlist JSON -> EDIF that OrCAD Capture imports as a drawn schematic.

What ends up in the .dsn
- Real symbols. Each class uses a standard symbol (bundled KiCad library:
  US zig-zag resistor, diode, NPN, gates, ...). Its rotation / mirror is the
  one that puts its pins where the wires met the part in the drawing, and
  matches detected pin roles (anode/cathode, base/collector/emitter), so pin
  numbers are the standard ones (diode K=1 A=2, BJT B=1 C=2 E=3).
- Real wires. Every wire island of the drawing is routed again on OrCAD's
  100 mil grid, following the original path where it can (cheaper cost on
  grid points the drawn wire passed through). Different nets never touch:
  they may only cross at right angles mid-segment, which Capture does not
  connect. Branches of one net meet at segment ends, where Capture draws
  its junction dot.
- Ground and supply symbols become Capture power objects; Input / Output /
  Common style terminals become ports. Both are written the way Capture
  itself exports them (a cell with CELLTYPE "pagePort" and a top-level
  port; PORTTYPE "supply" for power), so the importer rebuilds real ones.
- A "PCB Footprint" property on every part (configs/footprints.yaml) and
  unique reference designators (unread ones are numbered on from the
  highest one read: R21, R22 ...; property C2D_AUTO_REF marks them).
- Where the router cannot reach a pin, that pin gets a short stub with a
  net alias; same-named aliases are connected by Capture, so connectivity
  never depends on the router.

Coordinates: "grid" units are 100 mil. Symbol templates are y-up (as in
EDIF); page grid coordinates are y-down, converted when written.
"""

from __future__ import annotations

import heapq
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import PROJECT_ROOT
from .edif import (G, PREFIX, TEXT_H, EdifReport, Names, Str, ascii_text, emit, pin_numbers, pin_order, pt,
                   split_pages)

SYMBOL_DIR = PROJECT_ROOT / "circuit2dsn" / "kicad_symbols"
FOOTPRINTS = PROJECT_ROOT / "configs" / "footprints.yaml"

CLASS_SYMBOL = {
    "resistor": ("Device", "R_US"), "capacitor": ("Device", "C"), "inductor": ("Device", "L"),
    "diode": ("Device", "D"), "led": ("Device", "LED"), "bjt": ("Transistor_BJT", "Q_NPN_BCE"),
    "mosfet": ("Transistor_FET", "Q_NMOS_GDS"), "opamp": ("Amplifier_Operational", "TL072"),
    "crystal": ("Device", "Crystal"), "fuse": ("Device", "Fuse"), "battery": ("Device", "Battery"),
    "buzzer": ("Device", "Buzzer"), "antenna": ("Device", "Antenna"),
    "transformer": ("Device", "Transformer_1P_1S"), "switch": ("Switch", "SW_Push"),
    "potentiometer": ("Device", "R_Potentiometer_US"), "thyristor": ("Device", "Q_SCR_AGK"),
    "triac": ("Device", "Q_Triac"), "varistor": ("Device", "Varistor"), "optocoupler": ("Isolator", "PC817"),
    "relay": ("Relay", "Relay_SPDT"), "speaker": ("Device", "Speaker"), "microphone": ("Device", "Microphone"),
    "motor": ("Motor", "Motor_DC"), "lamp": ("Device", "Lamp"), "meter": ("Device", "Voltmeter_DC"),
    "and_gate": ("74xx", "74LS08"), "or_gate": ("74xx", "74LS32"), "nand_gate": ("74xx", "74LS00"),
    "nor_gate": ("74xx", "74LS02"), "xor_gate": ("74xx", "74LS86"), "xnor_gate": ("4xxx", "4077"),
    "not_gate": ("74xx", "74LS04"), "buffer_gate": ("74xx", "74LS07"),
}
PNP = re.compile(r"^(2N(3906|2907|4403|5401|2905)|BC(55[6-9]|32[78]|807|857|858|859|860)|BD(13[68]|140)|"
                 r"TIP(3[02]|42|12[5-7]|147)|MJE(350|2955)|S8550|A1015|2SA\w+|MMBT3906|PNP)", re.I)
PMOS = re.compile(r"^(IRF9\w+|BSS84|AO340[17]|SI23\d\d|FQP27P06|NDS0605|PMOS|P-?CH)", re.I)
GLOBAL_CLASSES = {"ground", "power"}
PORT_CLASSES = {"port", "terminal"}
DROP_CLASSES = {"wire_hop", "no_connect", "net_tie"}
SOURCES = {"vsource", "isource"}
ORIENTS = ["R0", "R90", "R180", "R270", "MY", "MX", "MYR90", "MXR90"]
DIRS = [(1, 0), (0, 1), (-1, 0), (0, -1)]          # page grid, y down
OWNER = "Schematic_TSC"


# ---------------------------------------------------------------- templates

@dataclass
class TPin:
    number: str
    name: str
    x: int                      # connection point, grid, y up
    y: int
    dx: int                     # direction a wire leaves the pin (y up)
    dy: int
    inner: tuple = (0.0, 0.0)   # body end of the pin line
    inverted: bool = False
    power: bool = False         # OrCAD power pin (VCC/GND): Capture joins these across the page, so no wire is assigned to one


@dataclass
class Template:
    key: str
    lines: list = field(default_factory=list)       # [[(x, y), ...], ...] y up, grid
    circles: list = field(default_factory=list)     # [(cx, cy, r)]
    pins: list = field(default_factory=list)        # [TPin]
    generic: bool = False
    fp: str = ""                # OrCAD library parts: their own footprint,
    prefix: str = ""            # reference prefix,
    value: str = ""             # part name
    source: str = ""            # and library

    def box(self, with_pins=True):
        xs = [p[0] for ln in self.lines for p in ln] + [c[0] + s * c[2] for c in self.circles for s in (-1, 1)]
        ys = [p[1] for ln in self.lines for p in ln] + [c[1] + s * c[2] for c in self.circles for s in (-1, 1)]
        if with_pins:
            for p in self.pins:
                xs += [p.x, p.inner[0]]
                ys += [p.y, p.inner[1]]
        if not xs:
            return (0.0, 0.0, 0.0, 0.0)
        return (min(xs), min(ys), max(xs), max(ys))


def orient_xy(o: str, x: float, y: float) -> tuple[float, float]:
    """EDIF orientation (mirror first, then counter-clockwise rotation),
    checked against Capture's import of all eight."""
    if o.startswith("MX"):
        y = -y
    elif o.startswith("MY"):
        x = -x
    rot = o[2:] if o.startswith("M") else o
    for _ in range({"": 0, "R0": 0, "R90": 1, "R180": 2, "R270": 3}[rot]):
        x, y = -y, x
    return x, y


def _from_kicad(key: str, lib: str, name: str) -> Template:
    from .kicad_render import SymbolLib

    prims = _symlib().load(lib, name)
    mm = 1 / 2.54
    raw_pins = []
    tpl = Template(key)
    for pr in prims:
        if pr.kind == "pin":
            (x0, y0), (x1, y1) = pr.pts
            raw_pins.append((pr, x0 * mm, y0 * mm, x1 * mm, y1 * mm))
    # lattice offset that puts the pins' cross-axis coordinates on the grid
    best = None
    for ox in (0.0, 0.5):
        for oy in (0.0, 0.5):
            err = 0.0
            for _, x0, y0, x1, y1 in raw_pins:
                horizontal = abs(x0 - x1) >= abs(y0 - y1)
                c = (y0 + oy) if horizontal else (x0 + ox)
                err += abs(c - round(c))
            if best is None or err < best[0] - 1e-6:
                best = (err, ox, oy)
    _, ox, oy = best
    for pr in prims:
        if pr.kind == "line":
            tpl.lines.append([(x * mm + ox, y * mm + oy) for x, y in pr.pts])
        elif pr.kind == "circle":
            cx, cy = pr.pts[0]
            tpl.circles.append((cx * mm + ox, cy * mm + oy, pr.radius * mm))
    for pr, x0, y0, x1, y1 in raw_pins:
        x0, y0, x1, y1 = x0 + ox, y0 + oy, x1 + ox, y1 + oy
        dx, dy = x0 - x1, y0 - y1
        n = math.hypot(dx, dy) or 1.0
        dx, dy = int(round(dx / n)), int(round(dy / n))
        # along the pin: extend outward to the grid; across: nearest grid
        if dx:
            sx = math.ceil(x0 - 1e-6) if dx > 0 else math.floor(x0 + 1e-6)
            sy = round(y0)
            y1 += sy - y0
        else:
            sy = math.ceil(y0 - 1e-6) if dy > 0 else math.floor(y0 + 1e-6)
            sx = round(x0)
            x1 += sx - x0
        tpl.pins.append(TPin(pr.number or str(len(tpl.pins) + 1), pr.name, int(sx), int(sy), dx, dy,
                             (x1, y1), pr.inverted))
    return tpl


def _source(key: str, current: bool) -> Template:
    t = Template(key)
    t.circles.append((0.0, 0.0, 1.0))
    if current:
        t.lines += [[(0, -0.6), (0, 0.6)], [(-0.25, 0.3), (0, 0.6), (0.25, 0.3)]]
    else:
        t.lines += [[(-0.2, 0.5), (0.2, 0.5)], [(0, 0.3), (0, 0.7)], [(-0.2, -0.5), (0.2, -0.5)]]
    t.pins = [TPin("1", "+", 0, 2, 0, 1, (0.0, 1.0)), TPin("2", "-", 0, -2, 0, -1, (0.0, -1.0))]
    return t


@lru_cache(maxsize=1)
def _symlib():
    from .kicad_render import SymbolLib

    return SymbolLib(SYMBOL_DIR)


@lru_cache(maxsize=None)
def template_for(key: str) -> Template | None:
    if key == "vsource":
        return _source(key, False)
    if key == "isource":
        return _source(key, True)
    lib_name = CLASS_SYMBOL.get(key) or {"bjt_pnp": ("Transistor_BJT", "Q_PNP_BCE"),
                                         "mosfet_p": ("Transistor_FET", "Q_PMOS_GDS")}.get(key)
    if lib_name is None:
        return None
    try:
        return _from_kicad(key, *lib_name)
    except (KeyError, FileNotFoundError, ValueError):
        return None


def generic_template(key: str, pins_side: dict[str, tuple[str, int]], box_w: int, box_h: int,
                     numbers: dict[str, str], names: dict[str, str]) -> Template:
    """Box symbol for parts without a standard symbol (ICs, unknown parts):
    one pin per detected pin, on the side where its wire arrived."""
    t = Template(key, generic=True)
    t.lines.append([(0, 0), (box_w, 0), (box_w, -box_h), (0, -box_h), (0, 0)])
    for pid, (side, pos) in pins_side.items():
        if side == "L":
            x, y, dx, dy, inner = -1, -pos, -1, 0, (0, -pos)
        elif side == "R":
            x, y, dx, dy, inner = box_w + 1, -pos, 1, 0, (box_w, -pos)
        elif side == "T":
            x, y, dx, dy, inner = pos, 1, 0, 1, (pos, 0)
        else:
            x, y, dx, dy, inner = pos, -box_h - 1, 0, -1, (pos, -box_h)
        t.pins.append(TPin(numbers[pid], names.get(pid, ""), x, y, dx, dy, inner))
    return t


# power objects and ports, drawn like Capture's own CAPSYM symbols; hotspot at (0, 0)
SPECIAL = {
    "GND": ([[(-1, 0), (1, 0)], [(-0.7, -0.3), (0.7, -0.3)], [(-0.4, -0.6), (0.4, -0.6)],
             [(-0.1, -0.9), (0.1, -0.9)]], (0, 1)),
    "VCC_BAR": ([[(0, 0), (0, 1)], [(-0.5, 1), (0.5, 1)]], (0, -1)),
    "PORT_IN": ([[(0, 0), (0.5, 0.5), (3, 0.5), (3, -0.5), (0.5, -0.5), (0, 0)]], (-1, 0)),
    "PORT_OUT": ([[(0, 0.5), (2.5, 0.5), (3, 0), (2.5, -0.5), (0, -0.5), (0, 0.5)]], (-1, 0)),
    "PORT_BI": ([[(0, 0), (0.5, 0.5), (2.5, 0.5), (3, 0), (2.5, -0.5), (0.5, -0.5), (0, 0)]], (-1, 0)),
}


def special_template(key: str) -> Template:
    lines, (dx, dy) = SPECIAL[key]
    t = Template(key, [list(map(tuple, ln)) for ln in lines])
    t.pins = [TPin("1", key, 0, 0, dx, dy, (0.0, 0.0))]
    return t


# ---------------------------------------------------------------- configuration

@lru_cache(maxsize=1)
def footprint_table() -> dict:
    try:
        import yaml

        return yaml.safe_load(FOOTPRINTS.read_text(encoding="utf-8")) or {}
    except (OSError, ImportError, ValueError):
        return {}


def footprint_for(cls: str, n_pins: int) -> tuple[str, bool]:
    """(footprint, sim_only)."""
    tab = footprint_table()
    if cls in set(tab.get("sim_only", [])):
        return "", True
    fp = (tab.get("footprints") or {}).get(cls) or tab.get("default", "")
    return str(fp).replace("{pins}", str(max(n_pins, 1))), False


# ---------------------------------------------------------------- placement

@dataclass
class Placed:
    kind: str                       # part / global / port
    part: dict
    tpl: Template
    orient: str = "R0"
    ox: int = 0                     # origin, page grid (y down)
    oy: int = 0
    pin_of: dict = field(default_factory=dict)      # detected pin id -> template pin index
    ident: str = ""
    cell: str = ""
    refdes: str = ""
    name: str = ""                  # global / port name
    port_dir: str = "INOUT"

    def to_page(self, x: float, y: float) -> tuple[float, float]:
        rx, ry = orient_xy(self.orient, x, y)
        return self.ox + rx, self.oy - ry

    def pin_page(self, i: int) -> tuple[int, int]:
        p = self.tpl.pins[i]
        x, y = self.to_page(p.x, p.y)
        return int(round(x)), int(round(y))

    def pin_dir(self, i: int) -> tuple[int, int]:
        p = self.tpl.pins[i]
        dx, dy = orient_xy(self.orient, p.dx, p.dy)
        return int(round(dx)), int(round(-dy))

    def bbox(self) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = self.tpl.box()
        cs = [self.to_page(x, y) for x in (x1, x2) for y in (y1, y2)]
        return (min(c[0] for c in cs), min(c[1] for c in cs), max(c[0] for c in cs), max(c[1] for c in cs))

    def body_points(self) -> set:
        """Grid points covered by the symbol (pin ends excluded)."""
        x1, y1, x2, y2 = self.bbox()
        pins = {self.pin_page(i) for i in range(len(self.tpl.pins))}
        return {(x, y) for x in range(math.ceil(x1 - 1e-6), math.floor(x2 + 1e-6) + 1)
                for y in range(math.ceil(y1 - 1e-6), math.floor(y2 + 1e-6) + 1)} - pins


def _match(tpl: Template, det: list[tuple[str, float, float, str | None]], box) -> tuple[str, dict, float]:
    """Best orientation and pin assignment: detected pins (id, x, y, role) in
    image px against the template pins, both relative to their box centre."""
    from scipy.optimize import linear_sum_assignment

    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    hw, hh = max((box[2] - box[0]) / 2, 1.0), max((box[3] - box[1]) / 2, 1.0)
    dv = np.array([((x - cx) / hw, -(y - cy) / hh) for _, x, y, _ in det]) if det else np.zeros((0, 2))
    best = ("R0", {}, float("inf"))
    for k, o in enumerate(ORIENTS):
        tp = np.array([orient_xy(o, p.x, p.y) for p in tpl.pins])
        x1, y1, x2, y2 = tpl.box()
        corners = np.array([orient_xy(o, x, y) for x in (x1, x2) for y in (y1, y2)])
        lo, hi = corners.min(0), corners.max(0)
        c, h = (lo + hi) / 2, np.maximum((hi - lo) / 2, 0.5)
        tv = (tp - c) / h
        if not len(det):
            return "R0", {}, 0.0
        cost = np.linalg.norm(dv[:, None, :] - tv[None, :, :], axis=2)
        for j, p in enumerate(tpl.pins):
            if p.power:
                cost[:, j] += 50.0
        for i, (_, _, _, role) in enumerate(det):
            if role:
                for j, p in enumerate(tpl.pins):
                    if p.name and p.name.upper() != role.upper():
                        cost[i, j] += 3.0
        r, cidx = linear_sum_assignment(cost)
        total = cost[r, cidx].sum() + 0.02 * k          # prefer plain rotations on ties
        if total < best[2]:
            best = (o, {det[i][0]: int(j) for i, j in zip(r, cidx)}, total)
    return best


def _match_cost(tpl: Template, det, box, o: str, assign: dict) -> float:
    """How well orientation o puts the assigned template pins where the
    detected pins are (same normalisation as _match)."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    hw, hh = max((box[2] - box[0]) / 2, 1.0), max((box[3] - box[1]) / 2, 1.0)
    x1, y1, x2, y2 = tpl.box()
    corners = np.array([orient_xy(o, x, y) for x in (x1, x2) for y in (y1, y2)])
    lo, hi = corners.min(0), corners.max(0)
    c, h = (lo + hi) / 2, np.maximum((hi - lo) / 2, 0.5)
    cost = 0.0
    for pid, x, y, _ in det:
        if pid in assign:
            p = tpl.pins[assign[pid]]
            tv = (np.array(orient_xy(o, p.x, p.y)) - c) / h
            cost += float(np.hypot(tv[0] - (x - cx) / hw, tv[1] + (y - cy) / hh))
    return cost + 0.02 * ORIENTS.index(o)


def _template_key(part: dict) -> str:
    cls = part["cls"]
    text = " ".join(str(v[0]) for v in (part.get("value"), part.get("designator")) if v) + " " + (part.get("name") or "")
    if cls == "bjt" and any(PNP.match(w) for w in text.split()):
        return "bjt_pnp"
    if cls == "mosfet" and any(PMOS.match(w) for w in text.split()):
        return "mosfet_p"
    return cls


def _side_of(part: dict, pin: dict) -> str:
    x1, y1, x2, y2 = part["box"]
    d = {"L": abs(pin["x"] - x1), "R": abs(pin["x"] - x2), "T": abs(pin["y"] - y1), "B": abs(pin["y"] - y2)}
    return min(d, key=d.get)


def _port_kind(name: str) -> tuple[str, str]:
    n = (name or "").upper()
    if re.search(r"\b(IN|INPUT|VIN|RX|RXD|SDI|MOSI)\b|^IN", n):
        return "PORT_IN", "INPUT"
    if re.search(r"\b(OUT|OUTPUT|VOUT|TX|TXD|SDO|MISO)\b|^OUT", n):
        return "PORT_OUT", "OUTPUT"
    return "PORT_BI", "INOUT"


def _orient_for(tpl: Template, want: tuple[int, int], prefer_mirror: bool = False) -> str:
    """Orientation whose pin-0 direction (page, y down) equals `want`."""
    order = ["R0", "MY", "R90", "R270", "R180", "MX"] if prefer_mirror else ["R0", "R90", "R270", "R180", "MY", "MX"]
    p = tpl.pins[0]
    for o in order:
        dx, dy = orient_xy(o, p.dx, p.dy)
        if (int(round(dx)), int(round(-dy))) == want:
            return o
    return "R0"


def place_page(sub: dict, rep: EdifReport) -> tuple[list[Placed], float]:
    pins_of = defaultdict(list)
    for pid, p in sub["pins"].items():
        pins_of[p["part"]].append(pid)
    items: list[Placed] = []
    net_size = {n["id"]: len(n["pins"]) for n in sub["nets"]}
    for part in sub["parts"]:
        cls = part["cls"]
        if cls in DROP_CLASSES:
            continue
        det = [(pid, sub["pins"][pid]["x"], sub["pins"][pid]["y"],
                (sub["pins"][pid].get("role") or [None, 0])[0]
                if (sub["pins"][pid].get("role") or [None, 0])[1] >= 0.5 else None)
               for pid in sorted(pins_of[part["id"]], key=pin_order)]
        if cls in GLOBAL_CLASSES or cls in PORT_CLASSES:
            if cls == "ground":
                key = "GND"
            elif cls == "power":
                key = "VCC_BAR"
            else:
                key = _port_kind(part.get("name") or "")[0]
            tpl = special_template(key)
            pl = Placed("global" if cls in GLOBAL_CLASSES else "port", part, tpl)
            if det:
                side = _side_of(part, sub["pins"][det[0][0]])
                want = {"L": (-1, 0), "R": (1, 0), "T": (0, -1), "B": (0, 1)}[side]
                pl.pin_of = {det[0][0]: 0}
                pl.orient = _orient_for(tpl, want, prefer_mirror=pl.kind == "port")
            items.append(pl)
            continue
        key = _template_key(part)
        tpl = template_for(key)
        chosen = part.get("orcad")          # a part picked from the OrCAD libraries in review
        if chosen:
            from . import orcad_parts

            ot = orcad_parts.template(chosen["file"], chosen["name"]) if orcad_parts.available() else None
            if ot is not None:
                tpl = ot
                # OCR'd pin names help the pin assignment when no role is known
                det = [(pid, x, y, role or ((sub["pins"][pid].get("name") or [None, 0])[0]
                                            if (sub["pins"][pid].get("name") or [None, 0])[1] >= 0.5 else None))
                       for pid, x, y, role in det]
                fixed = chosen.get("pin_map") or {}           # detected pin -> OrCAD pin number, set by hand
                if fixed:
                    num_ix = {p.number: i for i, p in enumerate(tpl.pins)}
                    assign = {pid: num_ix[n] for pid, n in fixed.items() if n in num_ix and pid in pins_of[part["id"]]}
                    if len(assign) == len(det):
                        o = min(ORIENTS, key=lambda o: _match_cost(tpl, det, part["box"], o, assign))
                        items.append(Placed("part", part, tpl, o, pin_of=assign))
                        continue
        if tpl is not None and len(det) > len(tpl.pins):
            # stray pins (a wire end that leads nowhere) must not turn a
            # resistor into a box: drop dangling ones, least confident first
            dangling = sorted((p for p in det if net_size.get(sub["pins"][p[0]]["net"], 0) <= 1),
                              key=lambda p: sub["pins"][p[0]].get("conf", 1.0))
            while len(det) > len(tpl.pins) and dangling:
                det.remove(dangling.pop(0))
        if tpl is not None and chosen:
            # an OrCAD part: wires only go to its signal pins (VCC/GND are joined by Capture itself).
            # More detected pins than signal pins = noise: keep the surest ones, lone pins go first.
            room = sum(not q.power for q in tpl.pins)
            if len(det) > room:
                def weak(p):
                    return (net_size.get(sub["pins"][p[0]]["net"], 0) <= 1, -sub["pins"][p[0]].get("conf", 1.0))
                gone = sorted(det, key=weak, reverse=True)[:len(det) - room]
                det = [p for p in det if p not in gone]
        if tpl is not None and not chosen and cls != "ic_box" and len(tpl.pins) < len(det) <= len(tpl.pins) + 5:
            # a few extra detected pins (pin-number text, a stray wire end): keep the real symbol,
            # drop the least trustworthy pins instead of turning the part into a box
            def doubt(p):                              # dropped first: lone pins, then the least sure
                return (net_size.get(sub["pins"][p[0]]["net"], 0) <= 1, -sub["pins"][p[0]].get("conf", 1.0))

            drop = sorted(det, key=doubt, reverse=True)[:len(det) - len(tpl.pins)]
            det = [p for p in det if p not in drop]
        if tpl is not None and len(det) <= len(tpl.pins):
            o, assign, _ = _match(tpl, det, part["box"])
            items.append(Placed("part", part, tpl, o, pin_of=assign))
        else:
            items.append(Placed("part", part, None))     # generic box, built once the scale is known
    s0 = _choose_scale(sub, items)
    for f in (1.0, 1.25, 1.6, 2.0, 2.6, 3.4):
        s = s0 * f
        _position(sub, items, s, pins_of)
        if _overlaps(items, margin=3) == 0 or _legalize(items):
            break
    _clear_exits(items, sub)
    rep.scale_units_per_px = max(rep.scale_units_per_px, round(s * G, 1))
    return items, s


def _choose_scale(sub: dict, items: list[Placed]) -> float:
    """Grid units per image px: drawn pin spacing -> template pin spacing."""
    ratios = []
    for it in items:
        if it.kind != "part" or it.tpl is None or len(it.pin_of) < 2:
            continue
        ids = list(it.pin_of)
        best = None
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                pa, pb = sub["pins"][ids[a]], sub["pins"][ids[b]]
                d = math.hypot(pa["x"] - pb["x"], pa["y"] - pb["y"])
                ta, tb = it.tpl.pins[it.pin_of[ids[a]]], it.tpl.pins[it.pin_of[ids[b]]]
                td = math.hypot(ta.x - tb.x, ta.y - tb.y)
                if d > 3 and (best is None or d > best[0]):
                    best = (d, td)
        if best:
            ratios.append(best[1] / best[0])
    s = float(np.median(ratios)) if ratios else 0.03
    longest = max(sub.get("width", 2000), sub.get("height", 2000))
    return float(np.clip(s, 4.0 / max(longest, 1), 400.0 / max(longest, 1)))


def _position(sub: dict, items: list[Placed], s: float, pins_of) -> None:
    for it in items:
        part = it.part
        if it.tpl is None or it.tpl.generic:
            it.tpl = _generic_for(sub, part, s, pins_of[part["id"]])
            it.pin_of = {pid: i for i, pid in enumerate(sorted(pins_of[part["id"]], key=pin_order))}
        x1, y1, x2, y2 = part["box"]
        if it.kind != "part" and it.pin_of:
            # power symbols and ports: hotspot where the wire met the symbol
            pid = next(iter(it.pin_of))
            gx, gy = sub["pins"][pid]["x"] * s, sub["pins"][pid]["y"] * s
            it.ox, it.oy = int(round(gx)), int(round(gy))
            continue
        cx, cy = (x1 + x2) / 2 * s, (y1 + y2) / 2 * s
        bx1, by1, bx2, by2 = it.tpl.box()
        corners = [orient_xy(it.orient, x, y) for x in (bx1, bx2) for y in (by1, by2)]
        tcx = (min(c[0] for c in corners) + max(c[0] for c in corners)) / 2
        tcy = (min(c[1] for c in corners) + max(c[1] for c in corners)) / 2
        it.ox, it.oy = int(round(cx - tcx)), int(round(cy + tcy))


def _generic_for(sub: dict, part: dict, s: float, pids: list[str]) -> Template:
    x1, y1, x2, y2 = part["box"]
    w = max(2, int(round((x2 - x1) * s)))
    h = max(2, int(round((y2 - y1) * s)))
    taken: dict[str, set] = defaultdict(set)
    sides: dict[str, tuple[str, int]] = {}
    for pid in sorted(pids, key=pin_order):
        p = sub["pins"][pid]
        side = _side_of(part, p)
        along = (p["y"] - y1) * s if side in "LR" else (p["x"] - x1) * s
        hi = h if side in "LR" else w
        want = min(max(int(round(along)), 0), hi)
        pos = None
        for step in range(0, 2 * hi + 2):
            for c in (want - step, want + step):
                if 0 <= c <= hi and c not in taken[side]:
                    pos = c
                    break
            if pos is not None:
                break
        if pos is None:                 # side full: grow the box
            if side in "LR":
                h += 1
                pos = h
            else:
                w += 1
                pos = w
        taken[side].add(pos)
        sides[pid] = (side, pos)
    numbers = pin_numbers(sub, sorted(pids, key=pin_order))
    # Capture splits a pin number at '+' and similar: keep [A-Za-z0-9_-], unique
    used = set()
    for pid in sorted(pids, key=pin_order):
        n = re.sub(r"[^A-Za-z0-9_\-]", "", numbers[pid]) or "1"
        base, k = n, 1
        while n.upper() in used:
            k += 1
            n = f"{base}_{k}"
        used.add(n.upper())
        numbers[pid] = n
    names = {}
    for pid in pids:
        nm = sub["pins"][pid].get("name")
        if nm and nm[1] >= 0.5 and ascii_text(nm[0]):
            names[pid] = ascii_text(nm[0])
    return generic_template(f"BOX_{part['id']}", sides, w, h, numbers, names)


def _overlaps(items: list[Placed], margin: float) -> int:
    boxes = [it.bbox() for it in items]
    n = 0
    order = sorted(range(len(boxes)), key=lambda i: boxes[i][0])
    for a_i, a in enumerate(order):
        ax1, ay1, ax2, ay2 = boxes[a]
        for b in order[a_i + 1:]:
            bx1, by1, bx2, by2 = boxes[b]
            if bx1 >= ax2 + margin:
                break
            if not (ay2 + margin <= by1 or by2 + margin <= ay1):
                n += 1
    return n


def _legalize(items: list[Placed], margin: int = 3, rounds: int = 600) -> bool:
    """Push overlapping symbols apart, each pair along the line between their
    centres on the axis needing the smaller move, sharing the move (power
    symbols and ports take all of it). True when nothing overlaps any more."""
    boxes = [list(it.bbox()) for it in items]

    def shift(i, dx, dy):
        items[i].ox += dx
        items[i].oy += dy
        b = boxes[i]
        boxes[i] = [b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy]

    for _ in range(rounds):
        moved = False
        order = sorted(range(len(items)), key=lambda i: boxes[i][0])
        for a_i, a in enumerate(order):
            for b in order[a_i + 1:]:
                ax1, ay1, ax2, ay2 = boxes[a]
                bx1, by1, bx2, by2 = boxes[b]
                if bx1 >= ax2 + margin + 40:
                    break
                ox = min(ax2, bx2) - max(ax1, bx1) + margin
                oy = min(ay2, by2) - max(ay1, by1) + margin
                if ox <= 0 or oy <= 0:
                    continue
                ca = ((ax1 + ax2) / 2, (ay1 + ay2) / 2)
                cb = ((bx1 + bx2) / 2, (by1 + by2) / 2)
                if ox <= oy:
                    d, sx, sy = ox, (1 if cb[0] >= ca[0] else -1), 0
                else:
                    d, sx, sy = oy, 0, (1 if cb[1] >= ca[1] else -1)
                d = int(math.ceil(d - 1e-9))
                pa, pb = items[a].kind == "part", items[b].kind == "part"
                if pa and not pb:
                    shift(b, sx * d, sy * d)
                elif pb and not pa:
                    shift(a, -sx * d, -sy * d)
                else:
                    h = (d + 1) // 2
                    shift(b, sx * h, sy * h)
                    shift(a, -sx * (d - h), -sy * (d - h))
                moved = True
        if not moved:
            return True
    return False


def _clear_exits(items: list[Placed], sub: dict, radius: int = 14) -> None:
    """Last resort after spreading: a symbol whose body still covers another
    symbol's pin exit (or whose own exit is covered) moves to the nearest
    spot where neither happens. Power symbols / ports and small parts move."""

    def size(it):
        x1, y1, x2, y2 = it.bbox()
        return (it.kind == "part", (x2 - x1) * (y2 - y1))

    for _ in range(3):
        body: dict = {}
        for k, it in enumerate(items):
            for q in it.body_points():
                body.setdefault(q, set()).add(k)
        exits: dict = {}
        for k, it in enumerate(items):
            for i in range(len(it.tpl.pins)):
                a, d = it.pin_page(i), it.pin_dir(i)
                exits.setdefault((a[0] + d[0], a[1] + d[1]), set()).add(k)
                exits.setdefault(a, set()).add(k)
        bad = set()
        for q, owners in exits.items():
            for k in body.get(q, ()):
                for o in owners:
                    if o != k:
                        bad.add(min((k, o), key=lambda j: size(items[j])))
        # pin points / exits of different nets on one grid point
        net_at: dict = {}
        for k, it in enumerate(items):
            for pid, i in it.pin_of.items():
                a, d = it.pin_page(i), it.pin_dir(i)
                for q in (a, (a[0] + d[0], a[1] + d[1])):
                    net_at.setdefault(q, []).append((k, sub["pins"][pid]["net"]))
        for q, lst in net_at.items():
            if len({n for _, n in lst}) > 1:
                ks = {k for k, _ in lst}
                if len(ks) > 1:
                    bad.add(min(ks, key=lambda j: size(items[j])))
        if not bad:
            return
        for k in sorted(bad, key=lambda j: size(items[j])):
            it = items[k]
            others_body = {q for q, ks in body.items() if k not in ks}
            others_exit = {q for q, ks in exits.items() if k not in ks}
            ox, oy = it.ox, it.oy
            best = None
            for r in range(1, radius + 1):
                for dx in range(-r, r + 1):
                    for dy in (-r, r) if abs(dx) != r else range(-r, r + 1):
                        it.ox, it.oy = ox + dx, oy + dy
                        mine_body = it.body_points()
                        mine_exit = set()
                        for i in range(len(it.tpl.pins)):
                            a, d = it.pin_page(i), it.pin_dir(i)
                            mine_exit |= {a, (a[0] + d[0], a[1] + d[1])}
                        if not (mine_body & others_exit) and not (mine_exit & others_body)                                 and not (mine_exit & others_exit):
                            best = (ox + dx, oy + dy)
                            break
                    if best:
                        break
                if best:
                    break
            it.ox, it.oy = best if best else (ox, oy)
            for q in [q for q, ks in body.items() if k in ks]:
                body[q].discard(k)
            for q in it.body_points():
                body.setdefault(q, set()).add(k)
            for q in [q for q, ks in exits.items() if k in ks]:
                exits[q].discard(k)
            for i in range(len(it.tpl.pins)):
                a, d = it.pin_page(i), it.pin_dir(i)
                exits.setdefault((a[0] + d[0], a[1] + d[1]), set()).add(k)
                exits.setdefault(a, set()).add(k)


# ---------------------------------------------------------------- routing

class Router:
    """A* on the 100 mil grid. Rules that keep Capture's connectivity exact:
    - grid edges carry one net; symbol bodies are blocked;
    - another net's wire may only be crossed at right angles where it runs
      straight (no bend, end or junction there), and the crossing wire runs
      straight through too;
    - a pin is entered / left only along its own direction, and the grid
      point just outside each pin is reserved for that pin's net."""

    def __init__(self, W: int, H: int):
        self.W, self.H = W, H
        self.blocked: set = set()
        self.owner: dict = {}          # node -> net key
        self.dirs: dict = defaultdict(set)
        self.edges: dict = {}
        self.pins: dict = {}           # node -> (net key, outward dir index)
        self.crossed: set = set()

    def add_pin(self, node, net, d):
        self.pins[node] = (net, d)
        self.owner[node] = net
        dx, dy = DIRS[d]
        out = (node[0] + dx, node[1] + dy)
        if out not in self.owner and out not in self.blocked:
            self.owner[out] = net
            self.dirs.setdefault(out, set())

    def _enter_ok(self, net, q, d, targets) -> tuple[bool, bool]:
        """(allowed, is_crossing) for moving into q in direction d."""
        x, y = q
        if not (0 <= x <= self.W and 0 <= y <= self.H):
            return False, False
        if q in self.pins:
            pnet, pd = self.pins[q]
            return (pnet == net and q in targets and (pd + 2) % 4 == d), False
        if q in self.blocked:
            return False, False
        o = self.owner.get(q)
        if o is None or o == net:
            return True, False
        dset = self.dirs.get(q, set())
        perp = {1, 3} if d in (0, 2) else {0, 2}
        if dset == perp and q not in self.crossed:
            return True, True
        return False, False

    def route(self, net, src, src_dir, targets: set, guide: set, bounds) -> list | None:
        x_lo, y_lo, x_hi, y_hi = bounds
        # never join where another net crosses: the new wire end would touch it
        targets = {t for t in targets if t not in self.crossed}
        if not targets:
            return None
        tx = [t[0] for t in targets]
        ty = [t[1] for t in targets]
        tbox = (min(tx), min(ty), max(tx), max(ty))

        def h(n):
            dx = max(tbox[0] - n[0], 0, n[0] - tbox[2])
            dy = max(tbox[1] - n[1], 0, n[1] - tbox[3])
            return 0.35 * (dx + dy)

        start = (src, src_dir if src_dir is not None else -1)
        g = {start: 0.0}
        prev = {}
        heap = [(h(src), 0.0, src, start[1], False)]
        expanded = 0
        while heap:
            f, cost, node, din, crossing = heapq.heappop(heap)
            state = (node, din)
            if cost > g.get(state, float("inf")) + 1e-9:
                continue
            if node in targets and node != src:
                path = [node]
                s = state
                while s in prev:
                    s = prev[s]
                    path.append(s[0])
                return path[::-1]
            expanded += 1
            if expanded > 150000:
                return None
            if node == src and src_dir is not None:
                moves = [src_dir]
            elif crossing:
                moves = [din]
            else:
                moves = [d for d in range(4) if din < 0 or d != (din + 2) % 4]
            for d in moves:
                dx, dy = DIRS[d]
                q = (node[0] + dx, node[1] + dy)
                if not (x_lo <= q[0] <= x_hi and y_lo <= q[1] <= y_hi):
                    continue
                e = (min(node, q), max(node, q))
                if e in self.edges:
                    continue
                ok, cross = self._enter_ok(net, q, d, targets)
                if not ok:
                    continue
                step = 0.35 if q in guide else 1.0
                if din >= 0 and d != din:
                    step += 1.5
                if cross:
                    step += 3.0
                nc = cost + step
                ns = (q, d)
                if nc < g.get(ns, float("inf")) - 1e-9:
                    g[ns] = nc
                    prev[ns] = state
                    heapq.heappush(heap, (nc + h(q), nc, q, d, cross))
        return None

    def commit(self, net, path) -> None:
        for a, b in zip(path, path[1:]):
            d = DIRS.index((b[0] - a[0], b[1] - a[1]))
            self.edges[(min(a, b), max(a, b))] = net
            for n, dd in ((a, d), (b, (d + 2) % 4)):
                o = self.owner.get(n)
                if o is not None and o != net:
                    self.crossed.add(n)          # straight crossing of another net's wire
                    continue
                self.owner[n] = net
                self.dirs[n].add(dd)


def _segments(paths: list[list]) -> list[tuple]:
    """Wire paths of one island -> straight segments split at bends, branch
    points and path ends (so every T meets at segment ends)."""
    adj = defaultdict(set)
    for path in paths:
        for a, b in zip(path, path[1:]):
            adj[a].add(b)
            adj[b].add(a)
    ends = {path[0] for path in paths} | {path[-1] for path in paths}
    brk = set(ends)
    for n, nb in adj.items():
        if len(nb) != 2:
            brk.add(n)
        else:
            a, b = list(nb)
            if (a[0] - n[0], a[1] - n[1]) != (n[0] - b[0], n[1] - b[1]):
                brk.add(n)
    segs, seen = [], set()
    for s in brk:
        for nb in adj[s]:
            if (s, nb) in seen:
                continue
            d = (nb[0] - s[0], nb[1] - s[1])
            cur = nb
            seen.add((s, nb))
            seen.add((nb, s))
            while cur not in brk:
                nxt = (cur[0] + d[0], cur[1] + d[1])
                seen.add((cur, nxt))
                seen.add((nxt, cur))
                cur = nxt
            segs.append((s, cur))
    return segs


# ---------------------------------------------------------------- writer

def _islands(sub: dict, net: dict, placed_pin: dict) -> list[list[str]]:
    """Pins of one net grouped by the wire island they were drawn on."""
    parent = {p: p for p in net["pins"] if p in placed_pin}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    by_group = defaultdict(list)
    contact = defaultdict(list)
    for pid in parent:
        g = sub["pins"][pid].get("group")
        if g is not None:
            by_group[g].append(pid)
        m = re.search(r"\.c(\d+)$", pid)
        if m:
            contact[m.group(1)].append(pid)
    for members in list(by_group.values()) + list(contact.values()):
        for m in members[1:]:
            parent[find(m)] = find(members[0])
    out = defaultdict(list)
    for pid in parent:
        out[find(pid)].append(pid)
    return [sorted(v, key=pin_order) for v in out.values()]


def _refdes(pages, rep) -> None:
    seen = defaultdict(int)
    for _, items in pages:
        for it in items:
            d = it.part.get("designator")
            if it.kind == "part" and d and d[1] >= 0.5:
                seen[ascii_text(d[0]).upper()] += 1
    used, top = set(), defaultdict(int)
    for _, items in pages:
        for it in items:
            if it.kind != "part":
                continue
            d = it.part.get("designator")
            text = ascii_text(d[0]).replace(" ", "") if d and d[1] >= 0.5 else ""
            if text and seen[text.upper()] == 1 and re.fullmatch(r"[A-Za-z]+\d+[A-Za-z]?", text):
                it.refdes = text
                used.add(text.upper())
                m = re.fullmatch(r"([A-Za-z]+)(\d+)", text)
                if m:
                    top[m.group(1).upper()] = max(top[m.group(1).upper()], int(m.group(2)))
    for _, items in pages:
        for it in items:
            if it.kind != "part" or it.refdes:
                continue
            prefix = (it.tpl.prefix or PREFIX.get(it.part["cls"], "X")).upper()
            k = top[prefix] + 1
            while f"{prefix}{k}" in used:
                k += 1
            top[prefix] = k
            it.refdes = f"{prefix}{k}"
            it.part = {**it.part, "_auto_ref": True}
            used.add(it.refdes)


def _net_names(netlist: dict, items_all: list[Placed]) -> dict[str, str]:
    """Final net names, unique. A net with a named power symbol or port takes
    that symbol's own spelling (so 'Input' stays 'Input')."""
    from .pipeline import norm_name

    sym_name = {}
    for it in items_all:
        if it.kind in ("global", "port") and it.part.get("name") and it.pin_of:
            pid = next(iter(it.pin_of))
            sym_name.setdefault(pid, it.part["name"])
    out, used = {}, set()
    for net in netlist["nets"]:
        raw = ascii_text(net.get("name") or "") or net["id"]
        for pid in net["pins"]:
            if pid in sym_name and norm_name(sym_name[pid]) == norm_name(raw):
                raw = ascii_text(sym_name[pid])
                break
        name, k = raw, 1
        while name.upper() in used:
            k += 1
            name = f"{raw}_{k}"
        used.add(name.upper())
        out[net["id"]] = name
    return out


def write_schematic(netlist: dict, design_name: str = "circuit") -> tuple[str, EdifReport]:
    rep = EdifReport()
    names = Names()
    lib, top = "CIRCUIT2DSN_LIB", "SCHEMATIC_TOP"
    names.make("", lib), names.make("", top)
    pages = []
    for n, sub in split_pages(netlist):
        items, s = place_page(sub, rep)
        pages.append((n, sub, items, s))
    _refdes([(n, items) for n, _, items, _ in pages], rep)
    net_name = _net_names(netlist, [it for _, _, items, _ in pages for it in items])
    auto_name = {nid: bool(re.fullmatch(r"N\d+(_\d+)?", nm)) for nid, nm in net_name.items()}

    cells, cell_of_tpl, page_nodes, top_ports = [], {}, [], {}
    port_used: set = set()
    special_cells: dict[str, str] = {}

    def cell_for(it: Placed) -> str:
        key = it.tpl.key
        if key in cell_of_tpl:
            return cell_of_tpl[key]
        cname = names.make("CELL", key)
        cell_of_tpl[key] = cname
        ports, impls = [], []
        for i, p in enumerate(it.tpl.pins):
            port = f"P{i + 1}"
            pname = ascii_text(p.name) if p.name and p.name not in ("~",) else p.number
            # Capture takes the port's display string as the pin NUMBER; the
            # pin name goes in a "Name" property (as Capture's own EDIF has it)
            ports.append(("port", ("rename", port, Str(pname or p.number)), ("direction", "INOUT"),
                          ("designator", Str(p.number)),
                          ("property", ("rename", "PACKAGEPORTNUMBERS", Str("PackagePortNumbers")),
                           ("string", Str(p.number)))))
            impls.append(("portImplementation", ("name", port),
                          ("connectLocation", ("figure", "GRAPHICS", ("dot", pt(p.x * G, p.y * G))))))
            ix, iy = p.inner
            if p.inverted:
                ux, uy = ix - p.x, iy - p.y
                n = math.hypot(ux, uy) or 1.0
                r = 0.25
                cxb, cyb = ix - ux / n * r, iy - uy / n * r
                impls.append(("figure", "PIN", ("path", ("pointList", pt(p.x * G, p.y * G),
                                                         pt((ix - ux / n * 2 * r) * G, (iy - uy / n * 2 * r) * G)))))
                impls.append(("figure", "BODY_FGP", ("circle", pt((cxb - r) * G, cyb * G), pt((cxb + r) * G, cyb * G))))
            elif (ix, iy) != (p.x, p.y):
                impls.append(("figure", "PIN", ("path", ("pointList", pt(p.x * G, p.y * G), pt(ix * G, iy * G)))))
        body = [("figure", "BODY_FGP", ("path", ("pointList", *[pt(x * G, y * G) for x, y in ln])))
                for ln in it.tpl.lines if len(ln) >= 2]
        body += [("figure", "BODY_FGP", ("circle", pt((cx - r) * G, cy * G), pt((cx + r) * G, cy * G)))
                 for cx, cy, r in it.tpl.circles]
        x1, y1, x2, y2 = it.tpl.box()
        prefix = it.tpl.prefix or PREFIX.get(it.part["cls"], "X")
        cells.append(("cell", cname, ("cellType", "GENERIC"),
                      ("view", "VIEW_1", ("viewType", "SCHEMATIC"),
                       ("interface", *ports,
                        ("designator", Str(prefix + "?")),
                        ("property", "VALUE", ("string", Str(""))),
                        ("symbol", ("boundingBox", ("rectangle", pt(x1 * G, y1 * G), pt(x2 * G, y2 * G))),
                         *body, *impls)))))
        return cname

    def special_cell(key: str, celltype: str) -> str:
        if key not in special_cells:
            cname = special_cells[key] = names.make("", key)
            lines = SPECIAL[key][0]
            group = {"GND": "POWEROBJECT", "VCC_BAR": "POWEROBJECT"}.get(key, "MODULEPORT")
            cells.append(("cell", cname, ("cellType", "GENERIC"),
                          ("property", "CELLTYPE", ("string", Str(celltype)), ("owner", Str(OWNER))),
                          ("view", cname, ("viewType", "GRAPHIC"), ("interface",),
                           ("contents", *[("figure", group, ("path", ("pointList", *[pt(x * G, y * G) for x, y in ln])))
                                          for ln in lines]))))
        return special_cells[key]

    for page_no, sub, items, s in pages:
        placed_pin = {}
        for it in items:
            for pid, i in it.pin_of.items():
                placed_pin[pid] = (it, i)
        xs = [v for it in items for v in (it.bbox()[0], it.bbox()[2])] or [0]
        ys = [v for it in items for v in (it.bbox()[1], it.bbox()[3])] or [0]
        sx, sy = 6 - int(math.floor(min(xs))), 6 - int(math.floor(min(ys)))
        for it in items:
            it.ox += sx
            it.oy += sy
        W = int(math.ceil(max(xs))) + sx + 8
        H = int(math.ceil(max(ys))) + sy + 8

        router = Router(W, H)
        for it in items:
            router.blocked |= it.body_points()
            # symbol pins with no wire in the drawing: a wire passing over
            # their end would connect them in Capture, so keep wires off
            used_pins = set(it.pin_of.values())
            for i in range(len(it.tpl.pins)):
                if i not in used_pins:
                    router.blocked.add(it.pin_page(i))
        nets_here = [net for net in sub["nets"] if any(p in placed_pin for p in net["pins"])]
        for net in nets_here:
            for pid in net["pins"]:
                if pid in placed_pin:
                    it, i = placed_pin[pid]
                    router.add_pin(it.pin_page(i), net["id"], DIRS.index(it.pin_dir(i)))

        # wire islands, routed shortest first
        guide_px = netlist.get("wires") if len(split_pages(netlist)) == 1 else sub.get("wires")
        jobs = []
        for net in nets_here:
            for isl in _islands(sub, net, placed_pin):
                pts_ = [placed_pin[p][0].pin_page(placed_pin[p][1]) for p in isl]
                span = (max(p[0] for p in pts_) - min(p[0] for p in pts_)) + (max(p[1] for p in pts_) - min(p[1] for p in pts_))
                jobs.append((span, net, isl))
        jobs.sort(key=lambda j: (j[0], len(j[2])))
        island_paths: dict[int, list] = {}
        failed: list[tuple] = []
        t0 = time.time()
        for k, (span, net, isl) in enumerate(jobs):
            guide = set()
            groups = {sub["pins"][p].get("group") for p in isl if sub["pins"][p].get("group") is not None}
            for gid in groups:
                for x, y in (guide_px or {}).get(str(gid), []):
                    gx, gy = int(round(x * s)) + sx, int(round(y * s)) + sy
                    guide.add((gx, gy))
            paths = []
            pins_pts = [(p, placed_pin[p][0].pin_page(placed_pin[p][1]), DIRS.index(placed_pin[p][0].pin_dir(placed_pin[p][1])))
                        for p in isl]
            if len(pins_pts) >= 2 and time.time() - t0 < 120:
                cx = np.mean([q[1][0] for q in pins_pts])
                cy = np.mean([q[1][1] for q in pins_pts])
                first = min(pins_pts, key=lambda q: abs(q[1][0] - cx) + abs(q[1][1] - cy))
                tree = {first[1]}
                rest = [q for q in pins_pts if q is not first]
                while rest:
                    rest.sort(key=lambda q: min(abs(q[1][0] - t[0]) + abs(q[1][1] - t[1]) for t in tree))
                    pid, node, d = rest.pop(0)
                    allx = [t[0] for t in tree] + [node[0]]
                    ally = [t[1] for t in tree] + [node[1]]
                    path = None
                    for pad in (10, 30, None):
                        b = (0, 0, W, H) if pad is None else (max(min(allx) - pad, 0), max(min(ally) - pad, 0),
                                                              min(max(allx) + pad, W), min(max(ally) + pad, H))
                        path = router.route(net["id"], node, d, tree, guide, b)
                        if path:
                            break
                    if path is None:
                        failed.append((net, pid))
                        continue
                    router.commit(net["id"], path)
                    paths.append(path)
                    tree |= set(path)
                # the first pin itself never failed; if nothing joined it, it gets a stub too
                if not paths and len(pins_pts) >= 2:
                    failed.append((net, first[0]))
            elif len(pins_pts) >= 2:
                failed += [(net, q[0]) for q in pins_pts]
            island_paths[k] = paths

        # an island that must carry the net's name (named net, or one of
        # several islands joined by name in the drawing) but has no wire
        # yet (a single pin) gets a stub to hang a visible alias on
        rep.unrouted += len(failed)
        n_isl = Counter(net["id"] for _, net, _ in jobs)
        failed_pins = {pid for _, pid in failed}
        for k, (span, net, isl) in enumerate(jobs):
            has_symbol = any(placed_pin[p][0].kind != "part" for p in isl)
            needs_name = (not auto_name[net["id"]] or n_isl[net["id"]] > 1) and not has_symbol
            if needs_name and not island_paths.get(k) and not any(p in failed_pins for p in isl):
                failed.append((net, isl[0]))
                failed_pins.add(isl[0])

        # stubs + aliases for pins the router could not reach
        stub_of = {}
        for net, pid in failed:
            it, i = placed_pin[pid]
            a = it.pin_page(i)
            dx, dy = it.pin_dir(i)
            path = [a]
            for _ in range(3):              # 1 grid stub, longer if the first point is taken
                b = (path[-1][0] + dx, path[-1][1] + dy)
                e = (min(path[-1], b), max(path[-1], b))
                if (router.owner.get(b) not in (None, net["id"]) or b in router.blocked or b in router.pins
                        or e in router.edges or b in router.crossed):
                    break
                path.append(b)
                if router.owner.get(b) is None or not router.dirs.get(b):
                    break
            end = path[-1]
            if len(path) >= 2 and (not router.dirs.get(end)
                                   or (router.owner.get(end) == net["id"] and end not in router.crossed)):
                router.commit(net["id"], path)
                stub_of[pid] = (path[0], path[-1])
                rep.stubs += 1
                continue
            rep.pins_without_stub.append(pid)

        def ex(gx):
            return int(round(gx)) * G

        def ey(gy):
            return (H - int(round(gy))) * G

        def disp(text, x, y, group="ATTRIBUTE", justify="LOWERLEFT"):
            return ("stringDisplay", Str(text),
                    ("display", ("figureGroupOverride", group, ("textHeight", TEXT_H), ("visible", ("true",))),
                     ("justify", justify), ("orientation", "R0"), ("origin", pt(x, y))))

        instances, specials = [], []
        port_ident: dict[str, str] = {}
        for it in items:
            bx1, by1, bx2, by2 = it.bbox()
            if it.kind == "part":
                it.cell = cell_for(it)
                it.ident = names.make("INST", it.part["id"])
                for pid, i in it.pin_of.items():
                    rep.pin_map[pid] = (it.ident, it.tpl.pins[i].number)
                rep.refdes[it.ident] = it.refdes
                value = it.part.get("value")
                vtext = ascii_text(value[0]) if value and value[1] >= 0.5 else (
                    ascii_text(it.tpl.value) if it.tpl.source and it.tpl.prefix.upper() not in ("R", "C", "L") else "")
                vertical = (by2 - by1) >= (bx2 - bx1)
                if vertical:
                    rx, ry, vx, vy, j = bx2 + 0.5, (by1 + by2) / 2 - 0.2, bx2 + 0.5, (by1 + by2) / 2 + 1.0, "LOWERLEFT"
                else:
                    rx, ry, vx, vy, j = (bx1 + bx2) / 2, by1 - 1.2, (bx1 + bx2) / 2, by2 + 1.6, "LOWERCENTER"
                fp, sim_only = footprint_for(it.part["cls"], len(it.tpl.pins))
                if it.tpl.source:                       # an OrCAD library part: its own footprint
                    fp, sim_only = it.tpl.fp or fp, False
                props = [("property", ("rename", "PCB_FOOTPRINT", Str("PCB Footprint")), ("string", Str(fp)))]
                if sim_only:
                    props.append(("property", ("rename", "PSPICEONLY", Str("PSpiceOnly")), ("string", Str("TRUE"))))
                if it.part.get("_auto_ref"):
                    props.append(("property", "C2D_AUTO_REF", ("string", Str("TRUE"))))
                if it.tpl.source:
                    props.append(("property", "ORCAD_LIBRARY", ("string", Str(it.tpl.source))))
                    props.append(("property", "ORCAD_PART", ("string", Str(it.tpl.value))))
                instances.append(("instance", ("rename", it.ident, Str(it.refdes)),
                                  ("viewRef", "VIEW_1", ("cellRef", it.cell, ("libraryRef", lib))),
                                  ("transform", ("orientation", it.orient), ("origin", pt(ex(it.ox), ey(it.oy)))),
                                  ("designator", disp(it.refdes, int(rx * G), int((H - ry) * G), justify=j)),
                                  ("property", "VALUE", ("string", disp(vtext, int(vx * G), int((H - vy) * G), justify=j))),
                                  *props,
                                  ("property", "CLASS", ("string", Str(it.part["cls"]))),
                                  ("property", "DETECT_CONF", ("string", Str(f"{it.part.get('conf', 1.0):.2f}"))),
                                  ("property", "SOURCE_ID", ("string", Str(it.part["id"])))))
                rep.parts += 1
            else:
                if not it.pin_of:
                    continue
                pid = next(iter(it.pin_of))
                net = next(n for n in sub["nets"] if pid in n["pins"])
                nm = net_name[net["id"]]
                if it.kind == "port":
                    key, direction = _port_kind(it.part.get("name") or nm)
                    if key != it.tpl.key:
                        want = it.pin_dir(0)
                        it.tpl = special_template(key)
                        it.orient = _orient_for(it.tpl, want, prefer_mirror=True)
                    celltype = "pagePort"
                else:
                    key, direction = it.tpl.key, "INOUT"
                    celltype = "pagePort"
                cname = special_cell(key, celltype)
                if nm.upper() not in port_ident:
                    ident = names.make("PORT", nm)
                    port_ident[nm.upper()] = ident
                    if it.kind == "global":
                        top_ports[ident] = ("port", ("rename", ident, Str(nm)),
                                            ("property", "IMPLICITPORTCLASS", ("string", Str(nm)), ("owner", Str(OWNER))),
                                            ("property", "PORTTYPE", ("string", Str("supply")), ("owner", Str(OWNER))))
                    else:
                        top_ports[ident] = ("port", ("rename", ident, Str(nm)), ("direction", direction))
                ident = port_ident[nm.upper()]
                it.ident = ident
                hx, hy = it.pin_page(0)
                tx, ty = it.to_page(3.4, 0) if it.kind == "port" else it.to_page(0, 1.3)
                show = it.kind == "port" or it.tpl.key != "GND"
                name_node = ("name", ident, ("display", "POWERTEXT" if it.kind == "global" else "ATTRIBUTE",
                                             ("justify", "CENTERLEFT" if it.kind == "port" else "LOWERCENTER"),
                                             ("origin", pt(ex(tx), ey(ty))))) if show else ("name", ident)
                specials.append(("portImplementation", name_node,
                                 ("instance", names.make("SYM", f"{key}_{it.part['id']}"),
                                  ("viewRef", cname, ("cellRef", cname, ("libraryRef", lib))),
                                  ("transform", ("orientation", it.orient), ("origin", pt(ex(hx), ey(hy))))),
                                 ("connectLocation", ("figure", "GRAPHICS", ("dot", pt(ex(hx), ey(hy)))))))
                rep.pin_map[pid] = ("TOP", nm)

        # nets: one EDIF net per wire island (+ one per stubbed pin)
        net_nodes = []
        emitted = set()
        for k, (span, net, isl) in enumerate(jobs):
            nid = net["id"]
            name = net_name[nid]
            emitted.add(nid)
            routed = [p for p in isl if p not in stub_of and p not in rep.pins_without_stub]
            refs = []
            for pid in routed:
                it, i = placed_pin[pid]
                if it.kind == "part":
                    refs.append(("portRef", f"P{i + 1}", ("instanceRef", it.ident)))
                elif it.ident not in port_used:
                    # a top-level port may belong to ONE EDIF net only; other
                    # islands with the same power symbol join it by net name
                    refs.append(("portRef", it.ident))
                    port_used.add(it.ident)
            segs = _segments(island_paths.get(k, []))
            figs = [("figure", "NET", ("path", ("pointList", pt(ex(a[0]), ey(a[1])), pt(ex(b[0]), ey(b[1])))))
                    for a, b in segs]
            n_islands = len(_islands(sub, net, placed_pin))
            has_symbol = any(placed_pin[p][0].kind != "part" for p in isl)
            label = (not auto_name[nid] or n_islands > 1) and not has_symbol and segs
            ident = names.make("NET", f"{nid}_{k}")
            if label:
                a, b = segs[0]
                lx, ly = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
                rn = ("rename", ident, disp(name, ex(lx), ey(ly) + G // 8, "LABEL", "LOWERCENTER"))
            else:
                rn = ("rename", ident, Str(name))
            if refs:
                net_nodes.append(("net", rn, ("joined", *refs), *figs))
            for pid in isl:
                if pid in stub_of or pid in rep.pins_without_stub:
                    it, i = placed_pin[pid]
                    if it.kind == "part":
                        ref = ("portRef", f"P{i + 1}", ("instanceRef", it.ident))
                    elif it.ident not in port_used:
                        ref = ("portRef", it.ident)
                        port_used.add(it.ident)
                    else:
                        continue
                    ident2 = names.make("NET", f"{nid}_{pid}")
                    if pid in stub_of:
                        a, b = stub_of[pid]
                        net_nodes.append(("net", ("rename", ident2, disp(name, ex(b[0]), ey(b[1]), "LABEL", "LOWERLEFT")),
                                          ("joined", ref),
                                          ("figure", "NET", ("path", ("pointList", pt(ex(a[0]), ey(a[1])), pt(ex(b[0]), ey(b[1])))))))
                    else:
                        net_nodes.append(("net", ("rename", ident2, Str(name)), ("joined", ref)))
        rep.nets += len(emitted)
        page_nodes.append(("page", f"PAGE_{page_no}", ("pageSize", ("rectangle", pt(0, 0), pt(W * G, H * G))),
                           *instances, *specials, *net_nodes))

    t = time.localtime()
    tree = ("edif", names.make("", design_name),
            ("edifVersion", 2, 0, 0), ("edifLevel", 0), ("keywordMap", ("keywordLevel", 0)),
            ("status", ("written", ("timestamp", t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec),
                        ("author", Str("circuit2dsn")), ("program", Str("circuit2dsn schematic.py"), ("version", Str("2.0"))))),
            ("library", lib, ("edifLevel", 0),
             ("technology",
              ("numberDefinition", ("scale", 1, ("e", 1, -8), ("unit", "DISTANCE")), ("gridMap", 1, 1)),
              ("figureGroup", "BODY_FGP", ("color", 50, 0, 50)),
              ("figureGroup", "ATTRIBUTE", ("color", 9, 25, 63), ("textHeight", TEXT_H)),
              ("figureGroup", "PIN", ("color", 50, 0, 50)),
              ("figureGroup", "NET", ("color", 0, 0, 100)),
              ("figureGroup", "LABEL", ("color", 9, 25, 63), ("textHeight", TEXT_H)),
              ("figureGroup", "POWEROBJECT", ("color", 100, 0, 0)),
              ("figureGroup", "POWERTEXT", ("color", 9, 25, 63), ("textHeight", TEXT_H)),
              ("figureGroup", "MODULEPORT", ("color", 0, 50, 0)),
              ("figureGroup", "GRAPHICS")),
             *cells,
             ("cell", top, ("cellType", "GENERIC"),
              ("view", "SCHEMATIC_VIEW", ("viewType", "SCHEMATIC"), ("interface", *top_ports.values()),
               ("contents", *page_nodes)))),
            ("design", ("rename", "DESIGN", Str(design_name)), ("cellRef", top, ("libraryRef", lib))))
    return "\n".join(emit(tree)) + "\n", rep
