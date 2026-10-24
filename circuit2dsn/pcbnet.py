"""Read PCBnet (Dataset300) annotations into plain structures.

PCBnet `edges` are visually connected wire networks: one edge = the pins that
are joined by drawn wire. Edges that carry the same net label (or the same
power symbol) form one electrical net, but merging by label is a text/OCR
job. The wire stage is scored against edges, the netlist stage against nets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .taxonomy import Taxonomy

IMAGE_EXTS = (".png", ".jpg", ".jpeg")


@dataclass
class GTPin:
    id: str
    comp: str
    tip: tuple[float, float]
    base: tuple[float, float]
    edge: str | None
    number_text: str | None = None     # text id of the pin number, when annotated
    name_text: str | None = None       # text id of the pin name, when annotated


@dataclass
class GTComponent:
    id: str
    raw_category: str
    cls: str | None
    box: tuple[float, float, float, float]       # includes pin stubs
    body: tuple[float, float, float, float]      # symbol body only
    pins: list[str] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)   # associated text ids


@dataclass
class GTText:
    id: str
    content: str
    kind: str          # designator / value / netlabel / pinName / pinNumber / ''
    box: tuple[float, float, float, float]


@dataclass
class GTDesign:
    uid: str
    image: Path
    width: int
    height: int
    components: dict[str, GTComponent]
    pins: dict[str, GTPin]
    edges: dict[str, list[str]]                  # edge id -> pin ids
    edge_labels: dict[str, str]                  # edge id -> net label text
    texts: dict[str, GTText]


def _box(d: dict) -> tuple[float, float, float, float]:
    return (d["x1"], d["y1"], d["x2"], d["y2"])


def load_design(folder: Path, tax: Taxonomy | None = None) -> GTDesign:
    folder = Path(folder)
    image = next(f for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTS)
    data = json.loads(next(folder.glob("*_CVJsonStd.json")).read_text(encoding="utf-8"))

    texts = {}
    for t in data["texts"].values():
        if t.get("state", "active") != "active":
            continue
        # kinds are spelled inconsistently ("netlabel" / "netLabel")
        texts[t["tid"]] = GTText(t["tid"], t.get("content") or "", (t.get("details") or "").lower(), _box(t["location"]))

    comps, pins = {}, {}
    for c in data["components"].values():
        if c.get("state", "active") != "active":
            continue
        m = tax.map("pcbnet", c["category"]) if tax else None
        comp = GTComponent(c["cpntId"], c["category"], m[0] if m else None,
                           _box(c["location"]), _box(c.get("realBBox") or c["location"]),
                           texts=[t for t in (c.get("associatedTexts") or {}) if t in texts])
        for pid, p in (c.get("pins") or {}).items():
            pins[pid] = GTPin(pid, comp.id, tuple(p["tip"]), tuple(p["base"]), None,
                              p.get("pinNumberByOCR") if p.get("pinNumberByOCR") in texts else None,
                              p.get("pinNameByOCR") if p.get("pinNameByOCR") in texts else None)
            comp.pins.append(pid)
        comps[comp.id] = comp

    # Some designs list one pin in several edges (a wire network annotated in
    # fragments). A pin sits on exactly one net, so edges sharing a pin are
    # merged; the merged edge keeps the first id.
    raw_edges = []
    for e in data["edges"].values():
        if e.get("state", "active") != "active":
            continue
        members = [pid for pid in e["connect"] if pid in pins]
        if members:
            raw_edges.append((e["edgeId"], members, e.get("netlabel")))
    parent = {eid: eid for eid, _, _ in raw_edges}

    def root(a: str) -> str:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    owner: dict[str, str] = {}
    for eid, members, _ in raw_edges:
        for pid in members:
            if pid in owner:
                parent[root(eid)] = root(owner[pid])
            else:
                owner[pid] = eid

    edges: dict[str, list[str]] = {}
    labels: dict[str, str] = {}
    for eid, members, lbl in raw_edges:
        r = root(eid)
        bucket = edges.setdefault(r, [])
        bucket.extend(p for p in members if p not in bucket)
        if lbl and lbl in texts and r not in labels:
            labels[r] = texts[lbl].content
    for r, members in edges.items():
        for pid in members:
            pins[pid].edge = r

    return GTDesign(folder.name, image, data["canvas"]["width"], data["canvas"]["height"],
                    comps, pins, edges, labels, texts)


def gt_nets(design: GTDesign) -> dict[str, str]:
    """pin id -> electrical net id: wire edges merged by ground symbols, the
    names of power/port symbols, and net-label text."""
    import re

    def norm(t: str) -> str:
        return re.sub(r"\s+", "", t).upper()

    parent = {e: e for e in design.edges}

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        parent[root(b)] = root(a)

    by_key: dict[str, list[str]] = {}
    for e, lbl in design.edge_labels.items():
        if lbl.strip():
            by_key.setdefault("L:" + norm(lbl), []).append(e)
    for c in design.components.values():
        edges = [design.pins[p].edge for p in c.pins if design.pins[p].edge]
        if c.cls == "ground":
            by_key.setdefault("GND", []).extend(edges)
        elif c.cls in ("power", "port"):
            for tid in c.texts:
                content = design.texts[tid].content.strip()
                if content and design.texts[tid].kind in ("netlabel", ""):
                    by_key.setdefault("L:" + norm(content), []).extend(edges)
                    break
    for members in by_key.values():
        for e in members[1:]:
            union(members[0], e)
    return {pid: root(p.edge) for pid, p in design.pins.items() if p.edge}
