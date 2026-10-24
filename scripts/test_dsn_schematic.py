"""End-to-end test of the drawn-schematic .dsn on PCBnet designs:

    netlist -> EDIF (real symbols, routed wires, power symbols, ports)
            -> OrCAD's own importer -> .dsn
            -> OrCAD's own netlist export of that .dsn -> same nets?

    python scripts/test_dsn_schematic.py --limit 30 --pdf

Netlists come from our pipeline run on the drawing with annotated parts and
text ("gt" mode), so the test is about the writer, not the detector.
Needs OrCAD installed (see circuit2dsn/orcad.py).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from circuit2dsn import orcad  # noqa: E402
from circuit2dsn.multipage import merge_pages  # noqa: E402
from circuit2dsn.pcbnet import load_design  # noqa: E402
from circuit2dsn.schematic import write_schematic  # noqa: E402
from circuit2dsn.taxonomy import Taxonomy  # noqa: E402
from scripts.eval_netlist import run_design  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--multipage", type=int, default=0, help="combine N designs into one multi-page document")
    ap.add_argument("--pdf", action="store_true", help="also print each .dsn to PDF (OrCAD)")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "dsn_test")
    args = ap.parse_args()
    if not orcad.available():
        sys.exit("OrCAD not found")
    tax = Taxonomy.load()
    splits = json.loads((ROOT / "data" / "yolo" / "detector" / "splits.json").read_text())
    uids = sorted(u.removeprefix("pcbnet_") for u in splits["pcbnet"][args.split])
    docs = [[u] for u in uids]
    if args.multipage > 1:
        docs = [uids[i:i + args.multipage] for i in range(0, len(uids), args.multipage)]
    if args.limit:
        docs = docs[:args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    tot = Counter()
    for doc in docs:
        uid = "+".join(doc)
        nls = [run_design(load_design(ROOT / "data" / "raw" / "Dataset300" / u, tax), "gt")[1] for u in doc]
        netlist = nls[0] if len(nls) == 1 else merge_pages(nls)
        d = args.out / uid
        d.mkdir(exist_ok=True)
        t0 = time.time()
        try:
            text, rep = write_schematic(netlist, f"D{uid}")
        except Exception as e:                      # noqa: BLE001 - report and continue
            print(f"{uid}: WRITER FAILED {type(e).__name__}: {e}", flush=True)
            tot["writer_failed"] += 1
            continue
        t_write = time.time() - t0
        edf = d / f"{uid}.edf"
        edf.write_text(text, encoding="ascii")
        t0 = time.time()
        res = orcad.edif_to_dsn(edf, d / f"D{uid}.DSN", expected=orcad.expected_nets(netlist, rep))
        t_import = time.time() - t0
        if args.pdf and res.ok:
            orcad.dsn_to_pdf(d / f"D{uid}.DSN", d / f"{uid}.pdf")
        tot["designs"] += 1
        tot["ok"] += bool(res.ok and res.errors == 0 and res.verified)
        tot["pins"] += res.pins_checked
        tot["mismatch"] += len(res.pins_mismatched)
        tot["parts"] += rep.parts
        tot["stubs"] += rep.stubs
        tot["unrouted"] += rep.unrouted
        tot["nostub"] += len(rep.pins_without_stub)
        tot["import_errors"] += max(res.errors, 0)
        print(f"{uid}: parts={rep.parts} nets={rep.nets} routed-stubs={rep.stubs} nostub={len(rep.pins_without_stub)} "
              f"| orcad ok={res.ok} err={res.errors} warn={res.warnings} verified={res.verified} "
              f"mismatch={len(res.pins_mismatched)}/{res.pins_checked} | {t_write:.1f}s + {t_import:.1f}s", flush=True)
        for m in res.pins_mismatched[:3]:
            print("     ", m)
    print(f"\n{tot['ok']}/{tot['designs']} designs verified by OrCAD; pin mismatches {tot['mismatch']}/{tot['pins']}; "
          f"import errors {tot['import_errors']}; parts {tot['parts']}; pins drawn as stub+alias {tot['stubs']} "
          f"(router could not reach {tot['unrouted']}; the rest are name labels as drawn; no stub {tot['nostub']}); writer failures {tot['writer_failed']}")


if __name__ == "__main__":
    main()
