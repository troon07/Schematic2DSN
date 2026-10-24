"""Convert a circuit2dsn netlist JSON to the industry NetList.xml format.

    python scripts/export_netlist_xml.py outputs/netlist/foo_netlist.json
    python scripts/export_netlist_xml.py foo_netlist.json --out foo_NetList.xml --board MYBOARD

The output follows the TD9 TestSequencer / SDDEMO schema: a <Nodes> section
with one <Node> per device pin, plus <PowerNets> and <UserNets> sections.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from circuit2dsn.netlist_xml import write_xml  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("netlist", type=Path, help="*_netlist.json from extract_netlist.py / review export")
    ap.add_argument("--out", type=Path, help="output .xml (default: <stem>_NetList.xml next to the json)")
    ap.add_argument("--application", default="TD9 TestSequencer")
    ap.add_argument("--board", default="SDDEMO")
    ap.add_argument("--version", default="FVER00001")
    args = ap.parse_args()

    nl = json.loads(args.netlist.read_text(encoding="utf-8"))
    out = args.out or args.netlist.with_name(args.netlist.stem.replace("_netlist", "") + "_NetList.xml")
    xml = write_xml(nl, str(out), application=args.application, board=args.board, version=args.version)
    print(f"wrote {out}")
    import re
    print(f"  nodes: {xml.count('<Node ')}  power nets: {xml.count('IsPowerNet=\"1\"')}  "
          f"user nets: {xml.count('IsPowerNet=\"0\"')}")


if __name__ == "__main__":
    main()