"""The installed OrCAD part libraries as a searchable catalogue + symbol templates.

The catalogue is built once by scripts/build_orcad_catalog.py (OrCAD exports
every .olb). A part picked in review is drawn in the .dsn with OrCAD's own
symbol graphics, pin names, pin numbers (section A), reference prefix and
PCB footprint; its routing and the OrCAD connectivity check work exactly as
for every other part.
"""

from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path

from . import PROJECT_ROOT

CATALOG = PROJECT_ROOT / "data" / "orcad_catalog"


def available() -> bool:
    return (CATALOG / "index.json").exists()


@lru_cache(maxsize=1)
def index() -> list[dict]:
    if not available():
        return []
    rows = json.loads((CATALOG / "index.json").read_text())
    for r in rows:
        r["_key"] = re.sub(r"[^a-z0-9]", "", r["name"].lower())
    return rows


@lru_cache(maxsize=64)
def _lib(file: str) -> dict:
    return json.loads((CATALOG / "parts" / f"{file}.json").read_text())


def part(file: str, name: str) -> dict | None:
    return _lib(file).get(name)


def search(query: str, limit: int = 60, pins: int | None = None) -> list[dict]:
    """Ranked matches: exact name, name prefix, name contains, library contains.
    With `pins`, parts with exactly that many pins come first."""
    q = query.strip().lower()
    qk = re.sub(r"[^a-z0-9]", "", q)
    words = [w for w in re.split(r"\s+", q) if w]
    out = []
    for r in index():
        lib = r["lib"].lower()
        if qk:
            if r["_key"] == qk:
                score = 0
            elif r["_key"].startswith(qk):
                score = 1
            elif qk in r["_key"]:
                score = 2
            elif all(w in lib or w in r["name"].lower() for w in words):
                score = 3
            else:
                continue
        else:
            score = 4
        if pins is not None:
            score += 0 if r["pins"] == pins else 0.5
        # prefer the plain libraries over PSpice / FPGA vendor variants on ties
        score += 0.1 * (lib.startswith(("pspice/", "fpga/")))
        out.append((score, len(r["name"]), r))
    out.sort(key=lambda t: (t[0], t[1], t[2]["name"]))
    return [{k: v for k, v in r.items() if k != "_key"} for _, _, r in out[:limit]]


def _arc(x1, y1, x2, y2, sx, sy, ex, ey, n=16):
    """Capture arc (ellipse box + start / end points, y down), counter-clockwise on screen."""
    cx, cy, rx, ry = (x1 + x2) / 2, (y1 + y2) / 2, max(abs(x2 - x1) / 2, 1e-6), max(abs(y2 - y1) / 2, 1e-6)
    a0 = math.atan2(-(sy - cy) / ry, (sx - cx) / rx)
    a1 = math.atan2(-(ey - cy) / ry, (ex - cx) / rx)
    span = (a1 - a0) % (2 * math.pi) or 2 * math.pi
    return [(cx + rx * math.cos(a0 + span * i / n), cy - ry * math.sin(a0 + span * i / n)) for i in range(n + 1)]


def template(file: str, name: str):
    """Catalogue part -> schematic.Template (grid units, y up)."""
    from .schematic import Template, TPin

    p = part(file, name)
    if p is None:
        return None
    row = next((r for r in index() if r["file"] == file and r["name"] == name), {})
    t = Template(f"ORCAD_{file}_{name}")
    t.fp, t.prefix, t.value, t.source = row.get("fp", ""), row.get("prefix", ""), name, row.get("lib", "")

    def xy(x, y):
        return (x / 10.0, -y / 10.0)

    for g in p["g"]:
        k = g[0]
        if k == "L":
            t.lines.append([xy(g[1], g[2]), xy(g[3], g[4])])
        elif k == "P":
            pts = [xy(x, y) for x, y in g[2]]
            t.lines.append(pts + ([pts[0]] if g[1] else []))
        elif k == "R":
            x1, y1, x2, y2 = g[1:5]
            t.lines.append([xy(x1, y1), xy(x2, y1), xy(x2, y2), xy(x1, y2), xy(x1, y1)])
        elif k == "E":
            x1, y1, x2, y2 = g[1:5]
            if abs((x2 - x1) - (y2 - y1)) <= 1:
                cx, cy = xy((x1 + x2) / 2, (y1 + y2) / 2)
                t.circles.append((cx, cy, abs(x2 - x1) / 20.0))
            else:
                pts = _arc(x1, y1, x2, y2, x2, (y1 + y2) / 2, x2, (y1 + y2) / 2, 24)
                t.lines.append([xy(x, y) for x, y in pts])
        elif k == "A":
            t.lines.append([xy(x, y) for x, y in _arc(*g[1:9])])
    used = set()
    bb = p.get("bbox") or [min(q[2] for q in p["pins"]), min(q[3] for q in p["pins"]),
                           max(q[2] for q in p["pins"]), max(q[3] for q in p["pins"])]
    for pname, num, hx, hy, sx, sy, ptype, dot in p["pins"]:
        dx, dy = hx - sx, -(hy - sy)
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            # zero-length pin: it points out of the side of the symbol it sits on
            side = min((abs(hx - bb[0]), (-1, 0)), (abs(hx - bb[2]), (1, 0)),
                       (abs(hy - bb[1]), (0, 1)), (abs(hy - bb[3]), (0, -1)))[1]
            dx, dy = side
        n = math.hypot(dx, dy) or 1.0
        dx, dy = int(round(dx / n)), int(round(dy / n))
        if dx and dy:                       # slanted pin: use its main axis
            dx, dy = (dx, 0) if abs(hx - sx) >= abs(hy - sy) else (0, dy)
        num = re.sub(r"[^A-Za-z0-9_\-]", "", str(num)) or str(len(t.pins) + 1)
        while num.upper() in used:
            num += "_"
        used.add(num.upper())
        t.pins.append(TPin(num, pname, int(round(hx / 10)), int(round(-hy / 10)), dx, dy,
                           (sx / 10.0, -sy / 10.0), bool(dot), ptype == 7))
    return t
