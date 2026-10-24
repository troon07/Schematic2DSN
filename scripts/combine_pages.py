"""Combine single-page netlists (e.g. reviewed pages) into one multi-page design + EDIF.

    python scripts/combine_pages.py outputs/review/doc_p1/export/reviewed_netlist.json \
        outputs/review/doc_p2/export/reviewed_netlist.json --out outputs/doc_all

Pages are numbered in the order given. Nets join across pages by name (GND,
power symbols, ports, net labels); --page-scope power joins only power,
ground and port names.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from circuit2dsn.multipage import merge_pages  # noqa: E402
from scripts.export_edif import export  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("netlists", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True, help="output path without extension")
    ap.add_argument("--page-scope", choices=["global", "power"], default="global")
    ap.add_argument("--style", choices=["labels", "joined"], default="labels")
    args = ap.parse_args()

    pages = [json.loads(p.read_text(encoding="utf-8")) for p in args.netlists]
    merged = merge_pages(pages, args.page_scope)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    js = args.out.with_name(args.out.name + "_netlist.json")
    js.write_text(json.dumps(merged, indent=1), encoding="utf-8")
    cross = [n for n in merged["nets"] if len(n.get("pages", [])) > 1]
    print(f"{len(pages)} pages, {len(merged['parts'])} parts, {len(cross)} nets joined across pages:")
    for n in cross:
        print(f"   {n['name']:<16} pages {n['pages']}  ({len(n['pins'])} pins)")
    r = export(merged, args.out.with_suffix(".edf"), args.out.name, args.style)
    print(f"wrote {js}\nwrote {r['file']}  self-check: {len(r['errors'])} errors, "
          f"{r['connectivity_mismatch']} mismatched pins, {r['shorted_pins']} shorts")


if __name__ == "__main__":
    main()
