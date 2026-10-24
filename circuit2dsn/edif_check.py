"""Parse an EDIF file back and check it the way OrCAD Capture will read it.

Checks:
- syntax: balanced parentheses, legal identifiers, ASCII strings
- references: every instance's cell exists; every portRef names a port of
  that instance's cell
- grid: every pin connection point and wire end is on the 100 mil grid
- graphical connectivity (Capture's model): a wire joins the pins whose
  connection points it touches; wires with the same net name join (labels)
Returns the connectivity as groups of (instance, pin number) so it can be
compared with the netlist that was written.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

G = 254000
IDENT = re.compile(r"[A-Za-z&][A-Za-z0-9_]*")


def tokenize(text: str):
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c in " \t\r\n":
            i += 1
        elif c in "()":
            yield c
            i += 1
        elif c == '"':
            j = text.index('"', i + 1)
            yield ("STR", text[i + 1:j])
            i = j + 1
        else:
            j = i
            while j < n and text[j] not in ' \t\r\n()"':
                j += 1
            yield text[i:j]
            i = j


def parse(text: str):
    stack, cur = [], []
    for tok in tokenize(text):
        if tok == "(":
            stack.append(cur)
            cur = []
        elif tok == ")":
            if not stack:
                raise ValueError("unbalanced ')'")
            done, cur = cur, stack.pop()
            cur.append(done)
        else:
            cur.append(tok)
    if stack:
        raise ValueError("unbalanced '(': file ends inside an expression")
    if len(cur) != 1:
        raise ValueError("expected exactly one top-level (edif ...) form")
    return cur[0]


def kids(node, key):
    return [c for c in node if isinstance(c, list) and c and c[0] == key]


def kid(node, key):
    k = kids(node, key)
    return k[0] if k else None


def name_of(node):
    """Identifier of `(keyword NAME ...)` or `(keyword (rename NAME "x") ...)`."""
    v = node[1]
    return v[1] if isinstance(v, list) and v[0] == "rename" else v


def label_of(node):
    v = node[1]
    if isinstance(v, list) and v[0] == "rename":
        s = v[2]
        if isinstance(s, list) and s[0] == "stringDisplay":
            s = s[1]
        return s[1] if isinstance(s, tuple) else s
    return v


def pts(node):
    out = []

    def walk(n):
        if isinstance(n, list):
            if n and n[0] == "pt":
                out.append((int(n[1]), int(n[2])))
            else:
                for c in n:
                    walk(c)
    walk(node)
    return out


@dataclass
class CheckResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    instances: int = 0
    nets: int = 0
    groups: list[set] = field(default_factory=list)      # graphical connectivity

    @property
    def ok(self) -> bool:
        return not self.errors


def check(text: str) -> CheckResult:
    res = CheckResult()
    try:
        tree = parse(text)
    except ValueError as e:
        res.errors.append(f"syntax: {e}")
        return res
    if tree[0] != "edif":
        res.errors.append("top-level form is not (edif ...)")
        return res
    for tok in re.findall(r'"([^"]*)"', text):
        if any(ord(c) > 126 for c in tok):
            res.errors.append(f"non-ASCII string: {tok[:30]!r}")

    cells = {}
    for lib in kids(tree, "library"):
        for cell in kids(lib, "cell"):
            cname = name_of(cell)
            if not IDENT.fullmatch(cname):
                res.errors.append(f"illegal cell name {cname}")
            view = kid(cell, "view")
            iface = kid(view, "interface") if view else None
            ports, conn = {}, {}
            for port in kids(iface or [], "port"):
                pname = name_of(port)
                d = kid(port, "designator")
                ports[pname] = d[1][1] if d else pname
            sym = kid(iface or [], "symbol")
            for impl in kids(sym or [], "portImplementation"):
                pname = impl[1][1] if isinstance(impl[1], list) else impl[1]
                p = pts(kid(impl, "connectLocation"))
                if p:
                    conn[pname] = p[0]
            for pname in ports:
                if pname not in conn and sym is not None:
                    res.errors.append(f"{cname}.{pname}: no connectLocation")
            cells[cname] = (ports, conn)

    top = None
    for lib in kids(tree, "library"):
        for cell in kids(lib, "cell"):
            view = kid(cell, "view")
            if view and kid(view, "contents"):
                top = kid(view, "contents")
    if top is None:
        res.errors.append("no schematic contents")
        return res
    pages = kids(top, "page") or [top]

    inst_cell, pin_at = {}, {}       # pin_at: (page index, point) -> pins
    for pg_i, page in enumerate(pages):
        for inst in kids(page, "instance"):
            iname = name_of(inst)
            cref = kid(kid(inst, "viewRef"), "cellRef")
            cname = cref[1] if cref else None
            if cname not in cells:
                res.errors.append(f"instance {iname}: unknown cell {cname}")
                continue
            inst_cell[iname] = cname
            tr = kid(inst, "transform")
            org = pts(kid(tr, "origin"))[0] if tr and kid(tr, "origin") else (0, 0)
            orient = kid(tr, "orientation")[1] if tr and kid(tr, "orientation") else "R0"
            if orient != "R0":
                res.warnings.append(f"{iname}: orientation {orient} not checked")
            for pname, (dx, dy) in cells[cname][1].items():
                p = (org[0] + dx, org[1] + dy)
                if p[0] % G or p[1] % G:
                    res.errors.append(f"{iname}.{pname} off grid at {p}")
                pin_at.setdefault((pg_i, p), []).append((iname, cells[cname][0][pname]))
            res.instances += 1

    # Union-find over pins: wires touching pins, and same-named nets (labels).
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        parent[find(a)] = find(b)

    for p, members in pin_at.items():
        for m in members:
            find(("pin", m))
        for m in members[1:]:
            union(("pin", members[0]), ("pin", m))
        if len(members) > 1:
            res.warnings.append(f"{len(members)} pins share grid point {p}: {members}")

    wire_points = defaultdict(list)
    for pg_i, page in enumerate(pages):
        for net in kids(page, "net"):
            res.nets += 1
            nid = ("net", label_of(net))
            find(nid)
            for ref in kids(kid(net, "joined") or [], "portRef"):
                iref = kid(ref, "instanceRef")
                iname = iref[1] if iref else None
                if iname not in inst_cell:
                    res.errors.append(f"net {name_of(net)}: portRef to unknown instance {iname}")
                elif ref[1] not in cells[inst_cell[iname]][0]:
                    res.errors.append(f"net {name_of(net)}: {iname} has no port {ref[1]}")
            for fig in kids(net, "figure"):
                path = pts(fig)
                for p in path:
                    if p[0] % G or p[1] % G:
                        res.errors.append(f"net {name_of(net)}: wire point off grid {p}")
                    wire_points[(pg_i, p)].append(nid)
                # Capture connects a wire to a pin only where they touch.
                touched = [m for p in path for m in pin_at.get((pg_i, p), [])]
                if not touched:
                    res.warnings.append(f"net {name_of(net)}: wire touches no pin")
                for m in touched:
                    union(("pin", m), nid)
    for p, nets in wire_points.items():
        names = {n[1] for n in nets}
        if len(names) > 1:
            res.errors.append(f"wires of different nets meet at {p}: {sorted(names)}")

    groups = defaultdict(set)
    for p in list(parent):
        if p[0] == "pin":
            groups[find(p)].add(p[1])
    res.groups = [g for g in groups.values()]
    return res
