"""Draw KiCad library symbols (.kicad_sym) with PIL, for synthetic training pages.

Reads unit 1, body style 1 (plus the shared unit 0) of a symbol, follows
"extends" to the parent symbol, and draws polylines, rectangles, circles,
arcs, beziers and pins (with the inverted-pin bubble). KiCad units are mm
with y pointing up.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw

from .edif_check import parse

NUM = re.compile(r"-?\d+(\.\d+)?")


def _atoms(node):
    return [x for x in node[1:] if not isinstance(x, list)]


def _kid(node, key):
    for x in node:
        if isinstance(x, list) and x and x[0] == key:
            return x
    return None


def _kids(node, key):
    return [x for x in node if isinstance(x, list) and x and x[0] == key]


def _s(v):
    return v[1] if isinstance(v, tuple) else v


@dataclass
class Prim:
    kind: str                       # line / rect / circle / arc / pin
    pts: list = field(default_factory=list)
    fill: str = "none"              # none / background / outline
    radius: float = 0.0
    inverted: bool = False
    number: str = ""                # pins only
    name: str = ""


def _fill(node) -> str:
    f = _kid(node, "fill")
    t = _kid(f, "type") if f else None
    return _s(t[1]) if t else "none"


def _xy(node):
    return (float(node[1]), float(node[2]))


def _arc_points(s, m, e, n=24):
    (x1, y1), (x2, y2), (x3, y3) = s, m, e
    d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(d) < 1e-9:
        return [s, e]
    ux = ((x1 * x1 + y1 * y1) * (y2 - y3) + (x2 * x2 + y2 * y2) * (y3 - y1) + (x3 * x3 + y3 * y3) * (y1 - y2)) / d
    uy = ((x1 * x1 + y1 * y1) * (x3 - x2) + (x2 * x2 + y2 * y2) * (x1 - x3) + (x3 * x3 + y3 * y3) * (x2 - x1)) / d
    r = math.hypot(x1 - ux, y1 - uy)
    a1, a2, a3 = (math.atan2(y - uy, x - ux) for x, y in (s, m, e))

    def between(a, b, c):           # is b on the arc from a to c going counter-clockwise?
        tau = 2 * math.pi
        return (b - a) % tau <= (c - a) % tau

    if between(a1, a2, a3):
        span = (a3 - a1) % (2 * math.pi)
    else:
        span = -((a1 - a3) % (2 * math.pi))
    return [(ux + r * math.cos(a1 + span * i / n), uy + r * math.sin(a1 + span * i / n)) for i in range(n + 1)]


def _bezier(pts, n=20):
    p0, p1, p2, p3 = pts[:4]
    out = []
    for i in range(n + 1):
        t = i / n
        a, b, c, d = (1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t * t * (1 - t), t ** 3
        out.append((a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0], a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1]))
    return out


class SymbolLib:
    """Loads symbols from an extracted kicad-symbols tree (<Lib>.kicad_symdir/<Name>.kicad_sym)."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.cache: dict = {}

    def _node(self, lib: str, name: str):
        path = self.root / f"{lib}.kicad_symdir" / f"{name}.kicad_sym"
        tree = parse(path.read_text(encoding="utf-8"))
        for sym in _kids(tree, "symbol"):
            if _s(sym[1]) == name:
                return sym
        raise KeyError(f"{lib}:{name}")

    def load(self, lib: str, name: str) -> list[Prim]:
        key = (lib, name)
        if key in self.cache:
            return self.cache[key]
        sym = self._node(lib, name)
        ext = _kid(sym, "extends")
        if ext is not None:
            prims = self.load(lib, _s(ext[1]))
            self.cache[key] = prims
            return prims
        base = _s(sym[1])
        prims: list[Prim] = []
        for unit in _kids(sym, "symbol"):
            uname = _s(unit[1])
            m = re.fullmatch(re.escape(base) + r"_(\d+)_(\d+)", uname)
            if not m or m.group(1) not in ("0", "1") or m.group(2) not in ("0", "1"):
                continue
            for el in unit[2:]:
                if not isinstance(el, list):
                    continue
                k = el[0]
                if k == "polyline":
                    pts = [_xy(p) for p in _kids(_kid(el, "pts"), "xy")]
                    prims.append(Prim("line", pts, _fill(el)))
                elif k == "rectangle":
                    (x1, y1), (x2, y2) = _xy(_kid(el, "start")), _xy(_kid(el, "end"))
                    prims.append(Prim("line", [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)], _fill(el)))
                elif k == "circle":
                    prims.append(Prim("circle", [_xy(_kid(el, "center"))], _fill(el), float(_kid(el, "radius")[1])))
                elif k == "arc":
                    pts = _arc_points(_xy(_kid(el, "start")), _xy(_kid(el, "mid")), _xy(_kid(el, "end")))
                    prims.append(Prim("line", pts, _fill(el)))
                elif k == "bezier":
                    pts = [_xy(p) for p in _kids(_kid(el, "pts"), "xy")]
                    if len(pts) >= 4:
                        prims.append(Prim("line", _bezier(pts), _fill(el)))
                elif k == "pin":
                    if _kid(el, "hide") is not None:
                        continue
                    at = _kid(el, "at")
                    x, y, rot = float(at[1]), float(at[2]), float(at[3]) if len(at) > 3 else 0.0
                    length = float(_kid(el, "length")[1])
                    dx, dy = math.cos(math.radians(rot)), math.sin(math.radians(rot))
                    style = _s(el[2]) if len(el) > 2 and not isinstance(el[2], list) else "line"
                    num, nam = _kid(el, "number"), _kid(el, "name")
                    prims.append(Prim("pin", [(x, y), (x + dx * length, y + dy * length)],
                                      inverted=style in ("inverted", "inverted_clock"),
                                      number=str(_s(num[1])) if num else "", name=str(_s(nam[1])) if nam else ""))
        self.cache[key] = prims
        return prims


@dataclass
class Drawn:
    image: Image.Image          # RGBA, transparent background
    pins: list                  # pin connection points in image px
    box: tuple                  # tight box of everything drawn (px, in image)


def render(prims: list[Prim], scale: float, rot: int = 0, mirror: bool = False,
           ink=(0, 0, 0), fill_bg=None, stroke: float = 2.0, pad: int = 6) -> Drawn:
    """scale: px per mm. rot: 0/90/180/270 degrees. mirror: flip x before rotating."""
    c, s = round(math.cos(math.radians(rot))), round(math.sin(math.radians(rot)))

    def tf(p):
        x, y = p
        if mirror:
            x = -x
        x, y = x * c - y * s, x * s + y * c
        return x * scale, -y * scale            # y up -> y down

    allpts = [tf(p) for pr in prims for p in pr.pts] + \
             [(tf(pr.pts[0])[0] + dx * pr.radius * scale, tf(pr.pts[0])[1] + dy * pr.radius * scale)
              for pr in prims if pr.kind == "circle" for dx, dy in ((1, 1), (-1, -1))]
    if not allpts:
        allpts = [(0, 0)]
    minx, miny = min(p[0] for p in allpts), min(p[1] for p in allpts)
    maxx, maxy = max(p[0] for p in allpts), max(p[1] for p in allpts)
    W, H = int(maxx - minx) + 2 * pad + 1, int(maxy - miny) + 2 * pad + 1
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    w = max(1, int(round(stroke)))

    def px(p):
        x, y = tf(p)
        return (x - minx + pad, y - miny + pad)

    ink_a = (*ink, 255)
    bg = (*fill_bg, 255) if fill_bg else None
    for pr in prims:
        if pr.kind == "line":
            pts = [px(p) for p in pr.pts]
            if pr.fill == "outline" and len(pts) >= 3:
                d.polygon(pts, fill=ink_a)
            elif pr.fill == "background" and bg and len(pts) >= 3:
                d.polygon(pts, fill=bg)
            if len(pts) >= 2:
                d.line(pts, fill=ink_a, width=w, joint="curve")
        elif pr.kind == "circle":
            cx, cy = px(pr.pts[0])
            r = pr.radius * scale
            f = ink_a if pr.fill == "outline" else (bg if pr.fill == "background" and bg else None)
            d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=ink_a, fill=f, width=w)
    pins = []
    for pr in prims:
        if pr.kind != "pin":
            continue
        a, b = px(pr.pts[0]), px(pr.pts[1])
        if pr.inverted:
            r = 0.635 * scale
            ux, uy = (b[0] - a[0]), (b[1] - a[1])
            n = math.hypot(ux, uy) or 1
            ux, uy = ux / n, uy / n
            cx, cy = b[0] - ux * r, b[1] - uy * r
            d.line([a, (b[0] - ux * 2 * r, b[1] - uy * 2 * r)], fill=ink_a, width=w)
            d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=ink_a, width=w)
        else:
            d.line([a, b], fill=ink_a, width=w)
        pins.append(a)
    bbox = img.getbbox() or (0, 0, W, H)
    return Drawn(img, pins, bbox)
