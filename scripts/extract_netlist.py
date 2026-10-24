"""Schematic page -> netlist JSON + overlay showing nets and doubtful links.

With the trained detector:
    python scripts/extract_netlist.py my_circuit.pdf

Demo before the detectors are trained (PCBnet parts + text boxes as the
"detectors"; text is read by OCR unless --gt-text):
    python scripts/extract_netlist.py --pcbnet 0002

Outputs in --out, per page:
    <name>_p<N>_netlist.json   parts, pins, nets (with conf + doubts), junctions
    <name>_p<N>_nets.png       each net its own colour; red ring = doubtful net,
                               magenta box = crossing/hop, black = junction dot
    <name>_p<N>.edf            EDIF for OrCAD Capture (File > Import Design > EDIF),
                               self-checked; see scripts/export_edif.py

Text needs the text detector (runs/text/weights/best.pt, trained like the
component detector). Without it nets are joined by wire and ground symbols
only; net labels and power names need recognised text.
"""

from __future__ import annotations

import argparse
import colorsys
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from circuit2dsn.pipeline import DOUBT_BELOW, Part, Text, extract_netlist  # noqa: E402
from circuit2dsn.multipage import merge_pages  # noqa: E402
from scripts.export_edif import export  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parent.parent


def draw_nets(page: Image.Image, result: dict) -> Image.Image:
    im = page.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    lw = max(2, round(max(im.size) / 1200))
    for p in result["parts"]:
        d.rectangle(p["box"], outline=(255, 160, 0), width=1)
    for k, net in enumerate(result["nets"]):
        r, g, b = colorsys.hsv_to_rgb((k * 0.618) % 1, 0.85, 0.8)
        col = (int(r * 255), int(g * 255), int(b * 255))
        doubtful = net["conf"] < DOUBT_BELOW
        for pid in net["pins"]:
            q = result["pins"][pid]
            rad = 3 + lw
            d.ellipse([q["x"] - rad, q["y"] - rad, q["x"] + rad, q["y"] + rad], fill=col)
            if doubtful:
                d.ellipse([q["x"] - rad - 3, q["y"] - rad - 3, q["x"] + rad + 3, q["y"] + rad + 3], outline=(230, 0, 0), width=2)
        if len(net["pins"]) > 1:
            q = result["pins"][net["pins"][0]]
            d.text((q["x"] + 6, q["y"] - 12), net["name"], fill=col)
    for j in result["junctions"]:
        col = {"crossing": (220, 0, 220), "hop": (220, 0, 220), "dot": (0, 0, 0)}.get(j["kind"])
        if col:
            d.rectangle([j["x"] - 4, j["y"] - 4, j["x"] + 4, j["y"] + 4], outline=col, width=1)
    return im


def from_pcbnet(uid: str, gt_text: bool):
    from circuit2dsn.pcbnet import load_design
    from circuit2dsn.taxonomy import Taxonomy
    from scripts.eval_netlist import design_texts

    design = load_design(ROOT / "data" / "raw" / "Dataset300" / uid, Taxonomy.load())
    page = Image.open(design.image).convert("RGB")
    parts = [Part(c.id, c.cls, c.box) for c in design.components.values() if c.cls]
    texts = design_texts(design, np.asarray(page), "gt" if gt_text else "ocr")
    return design.image, [(page, parts, texts)]


def from_detector(path: Path, weights: Path, text_weights: Path, dpi: int, conf: float):
    from circuit2dsn.detect import Detector, load_pages
    from circuit2dsn.textstage import ocr_inside_parts, ocr_page, read_texts

    det = Detector(weights, conf=conf)
    tdet = Detector(text_weights, conf=0.25) if text_weights.exists() else None
    if tdet is None:
        print(f"(no text detector at {text_weights}: using RapidOCR's own text finder)")
    pages = []
    for page in load_pages(path, dpi):
        parts = [Part(x.id, x.cls, (x.x1, x.y1, x.x2, x.y2), x.conf) for x in det.detect(page)]
        texts = []
        if tdet is not None:
            found = tdet.detect(page)
            texts = read_texts(np.asarray(page), [(x.x1, x.y1, x.x2, x.y2) for x in found],
                               [x.cls for x in found], role_confs=[x.conf for x in found])
        else:
            texts = ocr_page(np.asarray(page))      # no trained text model: RapidOCR finds the text
            texts += ocr_inside_parts(np.asarray(page), parts, texts)
        pages.append((page, parts, texts))
    return path, pages


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?", type=Path, help="PDF / PNG / JPG")
    ap.add_argument("--pcbnet", help="PCBnet design id (e.g. 0002): use its annotated parts + text")
    ap.add_argument("--weights", type=Path, default=ROOT / "runs" / "detector" / "weights" / "best.pt")
    ap.add_argument("--text-weights", type=Path, default=ROOT / "runs" / "text" / "weights" / "best.pt")
    ap.add_argument("--gt-text", action="store_true", help="with --pcbnet: annotated text content instead of OCR")
    ap.add_argument("--conf", type=float, default=0.35, help="detector confidence to keep a part")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "netlist")
    ap.add_argument("--page-scope", choices=["global", "power"], default="global",
                    help="join nets across pages by any name (global) or only power/ground/port names")
    ap.add_argument("--edif-style", choices=["labels", "joined", "none"], default="labels",
                    help="EDIF connectivity style (labels: every pin gets a named stub)")
    args = ap.parse_args()

    if args.pcbnet:
        src, pages = from_pcbnet(args.pcbnet, args.gt_text)
    else:
        if not args.input or not args.input.is_file():
            sys.exit("give an input file, or --pcbnet <id> for a demo")
        if not args.weights.exists():
            sys.exit(f"detector weights not found: {args.weights}\n"
                     "Train them (notebooks/train_detector_colab.ipynb) or try --pcbnet 0002")
        src, pages = from_detector(args.input, args.weights, args.text_weights, args.dpi, args.conf)

    args.out.mkdir(parents=True, exist_ok=True)
    from circuit2dsn.orientation import Orienter

    orienter = Orienter.available()       # None until scripts/train_orientation.py has run
    results = []
    for n, (page, parts, texts) in enumerate(pages, 1):
        result = extract_netlist(np.asarray(page), parts, texts, orienter=orienter)
        results.append(result)
        stem = f"{Path(src).stem}_p{n}"
        (args.out / f"{stem}_netlist.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
        draw_nets(page, result).save(args.out / f"{stem}_nets.png")
        nets = result["nets"]
        multi = [x for x in nets if len(x["pins"]) > 1]
        doubtful = [x for x in nets if x["conf"] < DOUBT_BELOW]
        print(f"page {n}: {len(parts)} parts, {len(result['pins'])} pins, {len(multi)} connected nets "
              f"({len(nets) - len(multi)} single-pin), {len(doubtful)} doubtful")
        for x in doubtful[:8]:
            print(f"   doubt {x['name']:<10} conf={x['conf']:.2f}  {x['doubts'][0]['what']}")
        print(f"   -> {args.out / stem}_nets.png")
        if args.edif_style != "none":
            r = export(result, args.out / f"{stem}.edf", stem, args.edif_style)
            status = "OK" if not r["errors"] and not r["connectivity_mismatch"] and not r["shorted_pins"] else "CHECK"
            print(f"   -> {r['file']}  (self-check {status}: {len(r['errors'])} errors, "
                  f"{r['connectivity_mismatch']} mismatched pins, {r['shorted_pins']} shorts)")

    if len(results) > 1:
        merged = merge_pages(results, args.page_scope)
        stem = f"{Path(src).stem}_all"
        (args.out / f"{stem}_netlist.json").write_text(json.dumps(merged, indent=1), encoding="utf-8")
        cross = [x for x in merged["nets"] if len(x.get("pages", [])) > 1]
        print(f"all {len(results)} pages: {len(merged['parts'])} parts, {len(cross)} nets joined across pages "
              f"({', '.join(x['name'] for x in cross[:8])}{'...' if len(cross) > 8 else ''})")
        if args.edif_style != "none":
            r = export(merged, args.out / f"{stem}.edf", stem, args.edif_style)
            print(f"   -> {r['file']}  ({len(results)} schematic pages; self-check "
                  f"{'OK' if not r['errors'] and not r['connectivity_mismatch'] else 'CHECK'})")


if __name__ == "__main__":
    main()
