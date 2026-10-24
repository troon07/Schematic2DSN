"""Netlist JSON -> EDIF for OrCAD Capture, with a self-check.

    python scripts/export_edif.py outputs/netlist/0002-Datasheet_p1_netlist.json
    python scripts/export_edif.py my_netlist.json --style joined

Then in OrCAD Capture: File > Import Design > EDIF tab > Open: the .edf,
Save As: the .dsn you want > OK.

The self-check parses the written file back and rebuilds connectivity the
way Capture does (wires touching pins + same-named labels); it must match
the netlist. Any mismatch, off-grid point or short is printed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from circuit2dsn.edif import write_edif  # noqa: E402
from circuit2dsn.edif_check import check  # noqa: E402


def expected_groups(netlist: dict, pin_map: dict) -> list[set]:
    """Nets as Capture should see them: only nets that the writer emitted."""
    out = []
    for net in netlist["nets"]:
        members = {pin_map[p] for p in net["pins"] if p in pin_map}
        if members:
            out.append(members)
    return out


def compare(expected: list[set], got: list[set]) -> tuple[int, int]:
    """(pins whose connections differ, pins checked). Singletons count too."""
    exp_of = {m: frozenset(g) for g in expected for m in g}
    got_of = {m: frozenset(g) for g in got for m in g}
    bad = sum(1 for m in exp_of if got_of.get(m, frozenset([m])) != exp_of[m])
    return bad, len(exp_of)


def export(netlist: dict, out: Path, name: str, style: str) -> dict:
    text, rep = write_edif(netlist, name, style)
    out.write_text(text, encoding="ascii")
    res = check(text)
    bad, total = compare(expected_groups(netlist, rep.pin_map), res.groups)
    return {"file": str(out), "parts": rep.parts, "nets": rep.nets, "stubs": rep.stubs,
            "pins_without_stub": len(rep.pins_without_stub), "shorted_pins": len(rep.shorted_pins),
            "errors": res.errors, "warnings": len(res.warnings), "connectivity_mismatch": bad, "pins": total,
            "scale_units_per_px": rep.scale_units_per_px}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("netlist", type=Path, help="*_netlist.json from extract_netlist.py")
    ap.add_argument("--style", choices=["labels", "joined"], default="labels")
    ap.add_argument("--out", type=Path, help="output .edf (default: next to the json)")
    args = ap.parse_args()

    netlist = json.loads(args.netlist.read_text(encoding="utf-8"))
    out = args.out or args.netlist.with_name(args.netlist.stem.replace("_netlist", "") + f"_{args.style}.edf")
    r = export(netlist, out, args.netlist.stem.replace("_netlist", ""), args.style)
    print(f"wrote {r['file']}")
    print(f"  {r['parts']} parts, {r['nets']} nets, {r['stubs']} wire stubs, scale {r['scale_units_per_px']:.0f} units/px")
    print(f"  self-check: {len(r['errors'])} errors, connectivity mismatches {r['connectivity_mismatch']}/{r['pins']} pins")
    for e in r["errors"][:10]:
        print("   ERROR", e)
    if r["shorted_pins"]:
        print(f"  WARNING: {r['shorted_pins']} pins of different nets landed on the same grid point")
    if r["pins_without_stub"]:
        print(f"  note: {r['pins_without_stub']} pins got no stub (would touch another net); they are listed in the net only")


if __name__ == "__main__":
    main()
