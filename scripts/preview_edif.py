"""Draw an EDIF schematic page to PNG (what Capture should show after import).

    python scripts/preview_edif.py outputs/netlist/0002-Datasheet_p1.edf

Draws part bodies, pins, wire stubs and net-name labels read back from the
file itself, so it also checks the geometry the writer produced.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from circuit2dsn.edif_check import G, kid, kids, label_of, parse, pts  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("edif", type=Path)
    ap.add_argument("--px-per-grid", type=int, default=12)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    tree = parse(args.edif.read_text(encoding="ascii"))
    cells, page = {}, None
    for lib in kids(tree, "library"):
        for cell in kids(lib, "cell"):
            view = kid(cell, "view")
            if kid(view, "contents"):
                page = kid(kid(view, "contents"), "page")
                continue
            sym = kid(kid(view, "interface"), "symbol")
            body = pts(kid(sym, "boundingBox")) if sym else [(0, 0), (G, -G)]
            pins = [pts(kid(i, "connectLocation"))[0] for i in kids(sym or [], "portImplementation")]
            cells[cell[1]] = (body, pins)
    size = pts(kid(page, "pageSize"))[1]
    s = args.px_per_grid / G
    W, H = int(size[0] * s) + 1, int(size[1] * s) + 1

    def xy(p):
        return p[0] * s, H - p[1] * s

    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    for inst in kids(page, "instance"):
        cname = kid(kid(inst, "viewRef"), "cellRef")[1]
        org = pts(kid(kid(inst, "transform"), "origin"))[0]
        (b0, b1), pins = cells[cname][0][:2], cells[cname][1]
        a, b = xy((org[0] + b0[0], org[1] + b0[1])), xy((org[0] + b1[0], org[1] + b1[1]))
        d.rectangle([min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])], outline=(150, 0, 150))
        for p in pins:
            x, y = xy((org[0] + p[0], org[1] + p[1]))
            d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(60, 60, 60))
        d.text((min(a[0], b[0]), min(a[1], b[1]) - 11), str(label_of(inst)), fill=(150, 0, 150))
    for net in kids(page, "net"):
        name = str(label_of(net))
        for fig in kids(net, "figure"):
            path = [xy(p) for p in pts(fig)]
            d.line(path, fill=(0, 90, 200), width=2)
            x, y = path[-1]
            d.text((x + 2, y - 10), name, fill=(0, 90, 200))
    out = args.out or args.edif.with_suffix(".preview.png")
    im.save(out)
    print(f"{out}  ({W}x{H} px)")


if __name__ == "__main__":
    main()
