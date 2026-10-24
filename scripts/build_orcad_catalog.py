"""Every part of the installed OrCAD Capture libraries -> searchable catalogue.

    python scripts/build_orcad_catalog.py            # ~30-60 min the first time
    python scripts/build_orcad_catalog.py --limit 3  # quick check
    python scripts/build_orcad_catalog.py --extra "D:/CompanyLibs"   # + your own .olb libraries

OrCAD exports each .olb with its own XMATIC_OLB2XML (data/orcad_catalog/xml,
skipped if already there, so a rerun continues). From the XML every part
keeps what the .dsn export needs: pins (name, number, position, direction),
the symbol drawing, reference prefix, PCB footprint and number of sections.

Output (data/orcad_catalog/):
  index.json           one row per part, for the search box
  parts/<lib>.json     symbol + pins per part, read when a part is exported
Coordinates are Capture's: 1/100 inch (10 = one 100 mil grid step), y down.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from circuit2dsn import orcad  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "orcad_catalog"

EXPORT = r"""
load [lindex $argv 0] DboTclWriteBasic
if {[catch {XMATIC_OLB2XML [lindex $argv 1] [lindex $argv 2]} e]} { puts "C2D_FAIL $e"; exit 3 }
exit 0
"""


def slug(rel: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", rel.rsplit(".", 1)[0]).strip("_").lower()


def export_xml(tools: Path, olb: Path, xml: Path) -> bool:
    if xml.exists() and xml.stat().st_size > 0:
        return True
    xml.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        try:
            orcad._run(tools, EXPORT, [olb.as_posix(), xml.as_posix()], Path(tmp), 600)
        except subprocess.TimeoutExpired:
            return False
    return xml.exists() and xml.stat().st_size > 0


def _i(d, k):
    try:
        return int(float(d.get(k, 0)))
    except ValueError:
        return 0


def graphics(view) -> list:
    g = []
    for e in view:
        d = e.find("Defn")
        a = d.attrib if d is not None else {}
        t = e.tag
        if t == "Line":
            g.append(["L", _i(a, "x1"), _i(a, "y1"), _i(a, "x2"), _i(a, "y2")])
        elif t in ("Polygon", "Polyline", "Bezier"):
            pts = [[_i(p.find("Defn").attrib, "x"), _i(p.find("Defn").attrib, "y")]
                   for p in e if p.tag in ("PolygonPoint", "PolylinePoint", "BezierPoint") and p.find("Defn") is not None]
            if len(pts) >= 2:
                g.append(["P", 1 if t == "Polygon" else 0, pts])
        elif t == "Rect":
            g.append(["R", _i(a, "x1"), _i(a, "y1"), _i(a, "x2"), _i(a, "y2")])
        elif t == "Ellipse":
            g.append(["E", _i(a, "x1"), _i(a, "y1"), _i(a, "x2"), _i(a, "y2")])
        elif t == "Arc":
            g.append(["A", _i(a, "x1"), _i(a, "y1"), _i(a, "x2"), _i(a, "y2"),
                      _i(a, "startX"), _i(a, "startY"), _i(a, "endX"), _i(a, "endY")])
        elif t == "CommentText" and a.get("name"):
            g.append(["T", _i(a, "locX"), _i(a, "locY"), a.get("name")[:40]])
    return g


def parse_lib(xml: Path, lib_rel: str):
    """-> (index rows, parts dict) for one library."""
    rows, parts = [], {}
    for _, pk in ET.iterparse(xml, events=("end",)):
        if pk.tag != "Package":
            continue
        d = pk.find("Defn")
        lp = pk.find("LibPart")
        view = lp.find("NormalView") if lp is not None else None
        if d is None or view is None:
            pk.clear()
            continue
        name = d.get("name", "")
        phys = lp.findall("PhysicalPart")
        nums = {}
        if phys:
            for pn in phys[0].findall("PinNumber"):
                pd = pn.find("Defn")
                nums[int(pd.get("position", 0))] = pd.get("number", "")
        pins = []
        for sp in view.findall("SymbolPinScalar"):
            a = sp.find("Defn").attrib
            pos = _i(a, "position")
            dot = (sp.find("IsDot/Defn") is not None and sp.find("IsDot/Defn").get("val") == "1")
            pins.append([a.get("name", ""), nums.get(pos, str(pos + 1)), _i(a, "hotptX"), _i(a, "hotptY"),
                         _i(a, "startX"), _i(a, "startY"), _i(a, "type"), 1 if dot else 0])
        if not pins:
            pk.clear()
            continue
        bb = view.find("SymbolBBox/Defn")
        parts[name] = {"pins": pins, "g": graphics(view),
                       "bbox": [_i(bb.attrib, k) for k in ("x1", "y1", "x2", "y2")] if bb is not None else None}
        rows.append({"name": name, "lib": lib_rel, "prefix": d.get("refdesPrefix", ""),
                     "fp": d.get("pcbFootprint", ""), "pins": len(pins), "sections": max(1, len(phys))})
        pk.clear()
    return rows, parts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--extra", nargs="*", default=[], type=Path,
                    help="more folders with .olb files (e.g. the company's own part libraries)")
    args = ap.parse_args()
    tools = orcad.find_tools()
    if tools is None:
        sys.exit("OrCAD not found")
    libdir = tools / "capture" / "library"
    olbs = sorted(p for p in libdir.rglob("*") if p.suffix.lower() == ".olb" and p.is_file())
    for extra in args.extra:
        olbs += sorted(p for p in Path(extra).rglob("*") if p.suffix.lower() == ".olb" and p.is_file())
    if args.limit:
        olbs = olbs[:args.limit]
    print(f"{len(olbs)} OrCAD libraries", flush=True)
    xmldir = OUT / "xml"

    def job(p):
        try:
            rel = p.relative_to(libdir).as_posix()
        except ValueError:                       # a company library outside the OrCAD install
            rel = "company/" + p.name
        ok = export_xml(tools, p, xmldir / (slug(rel) + ".xml"))
        return rel, ok

    done = 0
    results = []
    with ThreadPoolExecutor(args.workers) as ex:
        for rel, ok in ex.map(job, olbs):
            done += 1
            results.append((rel, ok))
            if done % 10 == 0 or not ok:
                print(f"  exported {done}/{len(olbs)}" + ("" if ok else f"  FAILED {rel}"), flush=True)
    (OUT / "parts").mkdir(parents=True, exist_ok=True)
    index = []
    for rel, ok in results:
        if not ok:
            continue
        rows, parts = parse_lib(xmldir / (slug(rel) + ".xml"), rel)
        for r in rows:
            r["file"] = slug(rel)
        index += rows
        (OUT / "parts" / f"{slug(rel)}.json").write_text(json.dumps(parts, separators=(",", ":")))
    (OUT / "index.json").write_text(json.dumps(index, separators=(",", ":")))
    print(f"catalogue: {len(index)} parts from {sum(ok for _, ok in results)} libraries -> {OUT}")


if __name__ == "__main__":
    main()
