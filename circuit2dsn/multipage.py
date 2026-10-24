"""Combine per-page netlists (pipeline output) into one multi-page design.

Nets join across pages by name, the flat-schematic convention:
- ground symbols: one GND net on every page
- power symbols and off-page ports: same name -> same net
- named nets (net labels): same name -> same net, unless scope="power"
Auto-numbered nets (N12, net7...) never join across pages.

Ids get a page prefix ("p2:C_5", "p2:C_5.1", "p2:net3"); each part keeps
its page number, and "pages" lists every page's size for the EDIF writer.
"""

from __future__ import annotations

import re

AUTO_NAME = re.compile(r"(N|net)\d+", re.I)


def _norm(name: str) -> str:
    return re.sub(r"\s+", "", name or "").upper()


def merge_pages(pages: list[dict], scope: str = "global") -> dict:
    if scope not in ("global", "power"):
        raise ValueError("scope must be 'global' or 'power'")
    out = {"parts": [], "pins": {}, "nets": [], "junctions": [], "texts": [], "pages": [],
           "stroke": max((p.get("stroke", 1.0) for p in pages), default=1.0)}
    groups: dict[str, list[dict]] = {}       # join key -> nets
    for n, nl in enumerate(pages, 1):
        pre = f"p{n}:"
        out["pages"].append({"page": n, "width": nl.get("width", 0), "height": nl.get("height", 0)})
        cls_of = {p["id"]: p["cls"] for p in nl["parts"]}
        for p in nl["parts"]:
            out["parts"].append({**p, "id": pre + p["id"], "page": n, "pins": [pre + q for q in p.get("pins", [])]})
        for pid, q in nl["pins"].items():
            out["pins"][pre + pid] = {**q, "part": pre + q["part"], "net": pre + q["net"]}
        for j in nl.get("junctions", []):
            out["junctions"].append({**j, "page": n})
        for t in nl.get("texts", []):
            out["texts"].append({**t, "page": n})
        for net in nl["nets"]:
            new = {**net, "id": pre + net["id"], "pins": [pre + q for q in net["pins"]], "pages": [n]}
            classes = {cls_of.get(nl["pins"][q]["part"]) for q in net["pins"]}
            name = net.get("name") or ""
            if "ground" in classes or _norm(name) == "GND":
                key = "GND"
            elif name and not AUTO_NAME.fullmatch(name) and (scope == "global" or classes & {"power", "port"}):
                key = "N:" + _norm(name)
            else:
                key = None
            if key:
                groups.setdefault(key, []).append(new)
            else:
                out["nets"].append(new)

    for key, nets in groups.items():
        if len(nets) == 1:
            out["nets"].append(nets[0])
            continue
        pages_of = sorted({p for nt in nets for p in nt["pages"]})
        first = nets[0]
        merged = {**first, "pins": [q for nt in nets for q in nt["pins"]], "pages": pages_of,
                  "conf": min(nt["conf"] for nt in nets),
                  "doubts": [d for nt in nets for d in nt.get("doubts", [])]}
        if len(pages_of) > 1:
            merged["doubts"].append({"what": f"joined across pages {pages_of} by name '{first.get('name')}'", "conf": 0.9})
        out["nets"].append(merged)
    for net in out["nets"]:
        for q in net["pins"]:
            out["pins"][q]["net"] = net["id"]
    return out
