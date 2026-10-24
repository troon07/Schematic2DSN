"""Find common circuit blocks in a netlist, so review can approve them as one unit.

Topology rules only (pin order of 2-pin parts is not known):
- bridge rectifier: 4 diodes forming a ring over 4 nets
- crystal + load caps: each crystal signal net has a capacitor to GND
- LED + series resistor: an LED and a resistor alone on a shared net
- voltage divider: two resistors share a net; one goes to GND, the other elsewhere
- RC filter: R and C share a net; the C goes to GND, the R does not
- decoupling caps: capacitors between a supply net and GND, grouped per supply
- pull-ups: resistors from a supply net to signal nets, grouped per supply
A part is used by at most one module (earlier rules win).
"""

from __future__ import annotations

import itertools
import re
from collections import defaultdict

SUPPLY = re.compile(r"(\+|-)?\d+(\.\d+)?V\d*|\d+V\d+|V(CC|DD|IN|BAT|BUS|REF|IO)\w*|.*V(CC|DD)\w*|.*\d+V\d*", re.I)


def _label(part: dict) -> str:
    d = part.get("designator")
    return d[0] if d and d[1] >= 0.5 else f"{part['cls']} {part['id']}"


def find_modules(netlist: dict) -> list[dict]:
    parts = {p["id"]: p for p in netlist["parts"]}
    nets = {n["id"]: n for n in netlist["nets"]}
    net_of = {pid: p["net"] for pid, p in netlist["pins"].items()}
    pins_of = defaultdict(list)
    for pid, p in netlist["pins"].items():
        pins_of[p["part"]].append(pid)
    nets_of = {cid: [net_of[p] for p in sorted(pins_of[cid])] for cid in parts}
    members = defaultdict(set)          # net -> parts on it
    for pid, p in netlist["pins"].items():
        members[p["net"]].add(p["part"])

    def name(nid):
        return (nets[nid].get("name") or "") if nid in nets else ""

    gnd = {nid for nid in nets if name(nid).upper() in ("GND", "AGND", "DGND", "PGND", "0", "VSS")}
    gnd |= {net_of[pid] for pid, p in netlist["pins"].items() if parts[p["part"]]["cls"] == "ground"}
    supply = {nid for nid in nets if nid not in gnd and SUPPLY.fullmatch(name(nid).replace(" ", ""))}
    supply |= {net_of[pid] for pid, p in netlist["pins"].items() if parts[p["part"]]["cls"] == "power"} - gnd

    used: set[str] = set()
    out: list[dict] = []

    def two_pin(cls):
        return [c for c, p in parts.items() if p["cls"] == cls and len(nets_of[c]) == 2 and c not in used]

    def add(kind, title, cids):
        cids = list(dict.fromkeys(cids))
        touched = sorted({n for c in cids for n in nets_of[c]})
        conf = min([parts[c].get("conf", 1.0) for c in cids] + [nets[n]["conf"] for n in touched if n in nets])
        out.append({"id": f"M{len(out) + 1}", "kind": kind, "title": title, "parts": cids, "nets": touched,
                    "conf": round(conf, 3)})
        used.update(cids)

    # bridge rectifier
    diodes = two_pin("diode")
    if 4 <= len(diodes) <= 40:
        for combo in itertools.combinations(diodes, 4):
            if any(c in used for c in combo):
                continue
            edges = [frozenset(nets_of[c]) for c in combo]
            if any(len(e) != 2 for e in edges):
                continue
            deg = defaultdict(int)
            for e in edges:
                for n in e:
                    deg[n] += 1
            if len(deg) == 4 and all(v == 2 for v in deg.values()):
                add("bridge_rectifier", "Bridge rectifier " + ", ".join(_label(parts[c]) for c in combo), combo)

    caps = lambda: two_pin("capacitor")  # noqa: E731
    # crystal + load caps
    for x, p in parts.items():
        if p["cls"] != "crystal" or x in used:
            continue
        loads = []
        for n in set(nets_of[x]) - gnd:
            for c in caps():
                if n in nets_of[c] and set(nets_of[c]) - {n} <= gnd and c not in loads:
                    loads.append(c)
        if loads:
            add("crystal", f"Crystal {_label(p)} with load caps " + ", ".join(_label(parts[c]) for c in loads), [x, *loads])

    # LED + series resistor
    for led in two_pin("led"):
        for n in nets_of[led]:
            rs = [r for r in two_pin("resistor") if n in nets_of[r] and members[n] == {led, r}]
            if rs and led not in used:
                add("led_resistor", f"LED {_label(parts[led])} with resistor {_label(parts[rs[0]])}", [led, rs[0]])

    # voltage divider
    for a, b in itertools.combinations(two_pin("resistor"), 2):
        if a in used or b in used:
            continue
        shared = set(nets_of[a]) & set(nets_of[b])
        if len(shared) != 1:
            continue
        mid = shared.pop()
        if mid in gnd or mid in supply:
            continue
        ra, rb = set(nets_of[a]) - {mid}, set(nets_of[b]) - {mid}
        if not ra or not rb:              # a resistor with both ends on one net: not a divider
            continue
        ea, eb = ra.pop(), rb.pop()
        if (ea in gnd) != (eb in gnd):
            add("divider", f"Voltage divider {_label(parts[a])} / {_label(parts[b])}", [a, b])

    # RC filter
    for r in two_pin("resistor"):
        if any(n in gnd or n in supply for n in nets_of[r]):
            continue
        for c in caps():
            shared = set(nets_of[r]) & set(nets_of[c])
            if len(shared) == 1 and (set(nets_of[c]) - shared) <= gnd and r not in used:
                add("rc_filter", f"RC filter {_label(parts[r])} + {_label(parts[c])}", [r, c])

    # decoupling caps per supply
    by_supply = defaultdict(list)
    for c in caps():
        a, b = nets_of[c]
        if (a in gnd and b in supply) or (b in gnd and a in supply):
            by_supply[b if a in gnd else a].append(c)
    for n, cs in by_supply.items():
        add("decoupling", f"Decoupling caps on {name(n) or n}: " + ", ".join(_label(parts[c]) for c in cs), cs)

    # pull-ups per supply
    by_supply = defaultdict(list)
    for r in two_pin("resistor"):
        a, b = nets_of[r]
        if (a in supply) != (b in supply) and a not in gnd and b not in gnd:
            by_supply[a if a in supply else b].append(r)
    for n, rs in by_supply.items():
        add("pullup", f"Pull-up resistors to {name(n) or n}: " + ", ".join(_label(parts[r]) for r in rs), rs)
    return out
