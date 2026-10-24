"""Round-trip test: PCBnet design -> netlist -> EDIF -> parse -> same nets?

    python scripts/test_edif_roundtrip.py            # test split, both styles

Uses annotated parts and text (fast); the EDIF writer is what is tested.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from circuit2dsn.pcbnet import load_design  # noqa: E402
from circuit2dsn.taxonomy import Taxonomy  # noqa: E402
from scripts.eval_netlist import run_design  # noqa: E402
from circuit2dsn.multipage import merge_pages  # noqa: E402
from scripts.export_edif import export  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test")
    ap.add_argument("--multipage", type=int, default=0,
                    help="combine N designs at a time into one multi-page document")
    args = ap.parse_args()
    tax = Taxonomy.load()
    splits = json.loads((ROOT / "data" / "yolo" / "detector" / "splits.json").read_text())
    uids = sorted(u.removeprefix("pcbnet_") for u in splits["pcbnet"][args.split])
    tot = {s: Counter() for s in ("labels", "joined")}
    tmp = Path(tempfile.mkdtemp())
    docs = [[u] for u in uids]
    if args.multipage > 1:
        docs = [uids[i:i + args.multipage] for i in range(0, len(uids), args.multipage)]
    for doc in docs:
        nls = [run_design(load_design(ROOT / "data" / "raw" / "Dataset300" / u, tax), "gt")[1] for u in doc]
        netlist = nls[0] if len(nls) == 1 else merge_pages(nls)
        uid = "+".join(doc)
        line = [uid]
        if len(nls) > 1:
            cross = sum(1 for n in netlist["nets"] if len(n.get("pages", [])) > 1)
            line.append(f"({len(nls)} pages, {cross} nets cross pages)")
        for style in ("labels", "joined"):
            r = export(netlist, tmp / f"{uid}_{style}.edf", uid, style)
            t = tot[style]
            t["designs"] += 1
            t["ok"] += not r["errors"] and r["connectivity_mismatch"] == 0
            t["errors"] += len(r["errors"])
            t["mismatch"] += r["connectivity_mismatch"]
            t["pins"] += r["pins"]
            t["no_stub"] += r["pins_without_stub"]
            t["shorted"] += r["shorted_pins"]
            line.append(f"{style}: err={len(r['errors'])} mismatch={r['connectivity_mismatch']}/{r['pins']} "
                        f"nostub={r['pins_without_stub']} short={r['shorted_pins']}")
            if r["errors"]:
                line.append(f"   first error: {r['errors'][0]}")
        print("  ".join(line), flush=True)
    for style, t in tot.items():
        print(f"\n{style}: {t['ok']}/{t['designs']} designs clean; errors {t['errors']}; "
              f"connectivity mismatches {t['mismatch']}/{t['pins']} pins; pins without stub {t['no_stub']}; "
              f"pins on another net's grid point {t['shorted']}")


if __name__ == "__main__":
    main()
