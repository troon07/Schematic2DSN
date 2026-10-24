"""Digitize-HCD (Mendeley Data 10.17632/rngcz5wtv8, CC BY 4.0) -> YOLO.

1,277 hand-drawn circuit diagrams (A4, 600 dpi scans) by 150+ people, 17
component classes as COCO boxes. Only the images and the component JSON are
read, straight out of the 2.1 GB zip with HTTP range requests (~250 MB
transferred), so the port-location crops in the same zip are never downloaded.

    python scripts/prepare_digitize_hcd.py                 # all images
    python scripts/prepare_digitize_hcd.py --limit 20      # quick check

Images are scaled to 2048 px on the long side. Finished images are skipped on
a rerun, so an interrupted run continues.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from circuit2dsn.taxonomy import Taxonomy  # noqa: E402
from remote_zip import HttpFile  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
# Mendeley's download link redirects here (public S3 object, supports ranges)
URL = "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/c4ec3d20-38ea-4f1d-96f9-cd504e998a6e"
BASE = "Digitize-HCD Dataset/Component Symbol and Text Label Data/"
CLASS = {"BJT-NPN": "bjt", "BJT-PNP": "bjt", "Capacitor": "capacitor", "Diode": "diode", "GND": "ground",
         "I-AC": "isource", "I-DC": "isource", "Inductor": "inductor", "MOSFET-N": "mosfet", "MOSFET-P": "mosfet",
         "Op-Amp": "opamp", "Resistor": "resistor", "V-AC": "vsource", "V-DC": "vsource",
         "V-DC (one port)": "power", "Wire Crossover": "wire_hop", "Zener Diode": "diode"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", default=URL, help="URL or local path of 'Digitize-HCD Dataset.zip'")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "yolo" / "digitize_hcd")
    ap.add_argument("--max-side", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    from PIL import Image

    tax = Taxonomy.load()
    src = HttpFile(args.zip) if str(args.zip).startswith("http") else open(args.zip, "rb")
    z = zipfile.ZipFile(src)
    coco = json.loads(z.read(BASE + "component_annotations.json"))
    cat = {c["id"]: CLASS[c["name"]] for c in coco["categories"]}
    anns: dict[int, list] = {}
    for a in coco["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    for sp in ("train", "val", "test"):
        (args.out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / sp).mkdir(parents=True, exist_ok=True)
    images = coco["images"][:args.limit] if args.limit else coco["images"]
    done = 0
    for k, im in enumerate(images):
        stem = "dhcd_" + Path(im["file_name"]).stem
        h = int(hashlib.md5(im["file_name"].encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        sp = "test" if h < 0.08 else "val" if h < 0.16 else "train"
        out_img = args.out / "images" / sp / f"{stem}.jpg"
        if out_img.exists():
            continue
        try:
            pic = Image.open(io.BytesIO(z.read(BASE + "Circuit Diagram Images/" + im["file_name"]))).convert("RGB")
        except KeyError:
            continue
        W, H = im["width"], im["height"]
        f = min(1.0, args.max_side / max(W, H))
        if f < 1.0:
            pic = pic.resize((round(W * f), round(H * f)), Image.LANCZOS)
        lines = []
        for a in anns.get(im["id"], []):
            x, y, w, hh = a["bbox"]
            if w < 2 or hh < 2:
                continue
            lines.append(f"{tax.index(cat[a['category_id']])} {(x + w / 2) / W:.6f} {(y + hh / 2) / H:.6f} "
                         f"{w / W:.6f} {hh / H:.6f}\n")
        (args.out / "labels" / sp / f"{stem}.txt").write_text("".join(lines))
        pic.save(out_img, quality=92)
        done += 1
        if done % 100 == 0:
            print(f"{k + 1}/{len(images)} images", flush=True)
    print(f"Digitize-HCD: {done} new images -> {args.out}")


if __name__ == "__main__":
    main()
