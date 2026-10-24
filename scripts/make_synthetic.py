"""Synthetic schematic pages from KiCad library symbols, with exact YOLO labels.

    python scripts/make_synthetic.py --pages 1500 --out data/yolo/synthetic

Each page: 8-24 symbols (new classes such as logic gates sampled more often),
random rotation / mirror / size, orthogonal wires between pins with junction
dots, designator / value / net-label text, and a random drawing style (KiCad
red-green, black, blue, beige paper) plus mild blur / JPEG noise. Labels are
the symbol's box including its pin stubs, like the PCBnet annotations.
Symbols come from circuit2dsn/kicad_symbols (bundled) or the full KiCad
library at data/kicad/kicad-symbols-master.
"""

from __future__ import annotations

import argparse
import io
import random
import sys
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from circuit2dsn.kicad_render import SymbolLib, render  # noqa: E402
from circuit2dsn.taxonomy import Taxonomy  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
# The symbols this script uses ship with the code (circuit2dsn/kicad_symbols);
# the full library (data/kicad/...) is used when present.
FULL_KICAD = ROOT / "data" / "kicad" / "kicad-symbols-master"
BUNDLED = ROOT / "circuit2dsn" / "kicad_symbols"

# class -> (weight, [(library, symbol), ...])
CATALOG = {
    "and_gate": (6, [("74xx", "74LS08"), ("74xx", "74LS11"), ("74xx", "74LS21"), ("4xxx", "4081"), ("4xxx", "4073"), ("74xGxx", "74LVC1G08")]),
    "or_gate": (6, [("74xx", "74LS32"), ("4xxx", "4071"), ("4xxx", "4075"), ("74xGxx", "74LVC1G32")]),
    "nand_gate": (6, [("74xx", "74LS00"), ("74xx", "74LS10"), ("74xx", "74LS20"), ("4xxx", "4011"), ("4xxx", "4023"), ("74xGxx", "74LVC1G00")]),
    "nor_gate": (6, [("74xx", "74LS02"), ("74xx", "74LS27"), ("4xxx", "4001"), ("4xxx", "4025"), ("74xGxx", "74LVC1G02")]),
    "xor_gate": (6, [("74xx", "74LS86"), ("4xxx", "4070"), ("74xGxx", "74LVC1G86")]),
    "xnor_gate": (5, [("4xxx", "4077")]),
    "not_gate": (6, [("74xx", "74LS04"), ("74xx", "74LS14"), ("4xxx", "4069"), ("4xxx", "40106"), ("74xGxx", "74LVC1G04")]),
    "buffer_gate": (5, [("74xx", "74LS07"), ("74xx", "74LS125"), ("4xxx", "4050"), ("74xGxx", "74LVC1G125")]),
    "potentiometer": (4, [("Device", "R_Potentiometer"), ("Device", "R_Potentiometer_US"), ("Device", "R_Potentiometer_Small")]),
    "thyristor": (4, [("Device", "Q_SCR_AGK"), ("Device", "Q_SCR_KAG")]),
    "triac": (4, [("Device", "Q_Triac")]),
    "varistor": (3, [("Device", "Varistor")]),
    "optocoupler": (4, [("Isolator", "PC817"), ("Isolator", "4N25"), ("Isolator", "TLP291")]),
    "relay": (4, [("Relay", "Relay_SPDT"), ("Relay", "G5LE-1"), ("Relay", "Relay_DPDT")]),
    "speaker": (3, [("Device", "Speaker"), ("Device", "Speaker_Crystal")]),
    "microphone": (3, [("Device", "Microphone"), ("Device", "Microphone_Condenser")]),
    "motor": (3, [("Motor", "Motor_DC"), ("Motor", "Motor_AC"), ("Motor", "Fan")]),
    "lamp": (3, [("Device", "Lamp"), ("Device", "Lamp_Neon")]),
    "meter": (3, [("Device", "Voltmeter_AC"), ("Device", "Voltmeter_DC"), ("Device", "Ammeter_AC"), ("Device", "Ammeter_DC")]),
    # existing classes keep appearing so the new ones are learned in context
    "resistor": (4, [("Device", "R"), ("Device", "R_US"), ("Device", "R_Small")]),
    "capacitor": (4, [("Device", "C"), ("Device", "C_Polarized"), ("Device", "C_Small"), ("Device", "C_Polarized_US")]),
    "inductor": (2, [("Device", "L"), ("Device", "L_Small")]),
    "diode": (3, [("Device", "D"), ("Device", "D_Zener"), ("Device", "D_Schottky")]),
    "led": (2, [("Device", "LED"), ("Device", "LED_Small")]),
    "bjt": (3, [("Transistor_BJT", "Q_NPN_BCE"), ("Transistor_BJT", "Q_PNP_BCE")]),
    "mosfet": (3, [("Transistor_FET", "Q_NMOS_GDS"), ("Transistor_FET", "Q_PMOS_GDS")]),
    "opamp": (3, [("Amplifier_Operational", "LM741"), ("Amplifier_Operational", "TL072")]),
    "ground": (3, [("power", "GND"), ("power", "GNDA"), ("power", "Earth")]),
    "power": (3, [("power", "VCC"), ("power", "+5V"), ("power", "+3V3")]),
    "crystal": (2, [("Device", "Crystal"), ("Device", "Crystal_Small")]),
    "fuse": (1, [("Device", "Fuse"), ("Device", "Fuse_Small")]),
    "battery": (1, [("Device", "Battery"), ("Device", "Battery_Cell")]),
    "buzzer": (1, [("Device", "Buzzer")]),
    "antenna": (1, [("Device", "Antenna")]),
    "transformer": (1, [("Device", "Transformer_1P_1S")]),
    "switch": (2, [("Switch", "SW_Push"), ("Switch", "SW_SPDT"), ("Switch", "SW_SPST")]),
}
PREFIX = {"resistor": "R", "potentiometer": "RV", "capacitor": "C", "inductor": "L", "diode": "D", "led": "D",
          "bjt": "Q", "mosfet": "Q", "thyristor": "Q", "triac": "Q", "opamp": "U", "relay": "K", "optocoupler": "U",
          "speaker": "LS", "microphone": "MK", "motor": "M", "lamp": "LA", "meter": "MES", "crystal": "Y",
          "fuse": "F", "battery": "BT", "buzzer": "BZ", "antenna": "AE", "transformer": "T", "switch": "SW",
          "varistor": "RV"}
STYLES = [  # (paper, body ink, wire ink, text ink, body fill)
    ((255, 255, 255), (0, 0, 0), (0, 0, 0), (0, 0, 0), None),
    ((245, 244, 239), (132, 0, 0), (0, 132, 0), (0, 100, 100), (255, 255, 194)),
    ((255, 255, 255), (0, 0, 160), (0, 0, 160), (0, 0, 160), None),
    ((255, 255, 255), (150, 30, 30), (40, 40, 140), (40, 40, 140), (255, 252, 230)),
    ((255, 255, 255), (60, 60, 60), (0, 0, 0), (0, 0, 0), (235, 235, 235)),
]
NETS = ["VCC", "GND", "SDA", "SCL", "CLK", "RST", "OUT", "IN", "EN", "A", "B", "Q", "Q_N", "DATA", "TX", "RX", "+5V", "3V3", "SIG", "LED_OUT"]
VALUES = ["10k", "4.7k", "100", "1M", "220", "100nF", "10uF", "1uF", "22pF", "1N4148", "BC547", "2N7000", "LM358", "12V", "5V", "8MHz"]


def font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def ensure_library() -> SymbolLib:
    if FULL_KICAD.exists():
        return SymbolLib(FULL_KICAD)
    if BUNDLED.exists():
        return SymbolLib(BUNDLED)
    z = ROOT / "kicad-symbols-master.zip"
    if not z.exists():
        sys.exit("no KiCad symbols: need circuit2dsn/kicad_symbols or kicad-symbols-master.zip")
    with zipfile.ZipFile(z) as zz:
        zz.extractall(FULL_KICAD.parent)
    return SymbolLib(FULL_KICAD)


def make_page(lib: SymbolLib, rng: random.Random, tax: Taxonomy):
    W, H = rng.randint(1100, 1800), rng.randint(800, 1300)
    paper, body, wire, text_ink, fill = rng.choice(STYLES)
    page = Image.new("RGB", (W, H), paper)
    d = ImageDraw.Draw(page)
    scale = rng.uniform(4.0, 8.5)                 # px per mm, same for the page (like a real sheet)
    stroke = max(1.0, scale * rng.uniform(0.18, 0.32))
    names = list(CATALOG)
    weights = [CATALOG[n][0] for n in names]
    placed, labels, pins_all = [], [], []
    target = rng.randint(8, 24)
    tries = 0
    counters: dict = {}
    while len(placed) < target and tries < target * 8:
        tries += 1
        cls = rng.choices(names, weights)[0]
        libname, sym = rng.choice(CATALOG[cls][1])
        try:
            prims = lib.load(libname, sym)
        except (KeyError, FileNotFoundError):
            continue
        g = render(prims, scale * rng.uniform(0.9, 1.1), rng.choice((0, 90, 180, 270)),
                   rng.random() < 0.3, body, fill, stroke)
        bw, bh = g.image.size
        if bw >= W - 40 or bh >= H - 40:
            continue
        x, y = rng.randint(20, W - bw - 20), rng.randint(20, H - bh - 20)
        box = (x + g.box[0], y + g.box[1], x + g.box[2], y + g.box[3])
        margin = 6 * scale
        if any(not (box[2] + margin < b[0] or b[2] + margin < box[0] or box[3] + margin < b[1] or b[3] + margin < box[1])
               for b, _ in placed):
            continue
        page.paste(g.image, (x, y), g.image)
        placed.append((box, cls))
        pins_all.extend((x + px, y + py, len(placed) - 1) for px, py in g.pins)
        labels.append((tax.index(cls), box))
        # designator + value text beside the part
        f = font(max(9, int(scale * 2.2)))
        if cls in PREFIX and rng.random() < 0.85:
            counters[cls] = counters.get(cls, 0) + 1
            tx, ty = (box[2] + 3, box[1]) if rng.random() < 0.6 else (box[0], box[1] - f.size - 3)
            d.text((tx, ty), f"{PREFIX[cls]}{counters[cls]}", fill=text_ink, font=f)
            if rng.random() < 0.5:
                d.text((tx, ty + f.size + 2), rng.choice(VALUES), fill=text_ink, font=f)
        elif cls.endswith("_gate") and rng.random() < 0.6:
            d.text((box[0], box[1] - f.size - 3), f"U{rng.randint(1, 20)}{rng.choice('ABCD')}", fill=text_ink, font=f)
    # wires: join pins of different parts with L-shaped orthogonal runs
    w = max(1, int(round(stroke)))
    rng.shuffle(pins_all)
    used = set()
    for i, (x1, y1, a) in enumerate(pins_all):
        if i in used:
            continue
        cands = [(j, p) for j, p in enumerate(pins_all) if j not in used and j != i and p[2] != a]
        if not cands or rng.random() < 0.15:
            if rng.random() < 0.5:                       # dangling stub with a net label
                L = rng.uniform(4, 12) * scale
                dx, dy = rng.choice(((1, 0), (-1, 0), (0, 1), (0, -1)))
                x2, y2 = x1 + dx * L, y1 + dy * L
                d.line([(x1, y1), (x2, y2)], fill=wire, width=w)
                f = font(max(9, int(scale * 2.0)))
                d.text((x2 + 2, y2 - f.size), rng.choice(NETS), fill=text_ink, font=f)
            used.add(i)
            continue
        j, (x2, y2, _) = min(cands, key=lambda c: abs(c[1][0] - x1) + abs(c[1][1] - y1) + rng.uniform(0, 200))
        used.update((i, j))
        if rng.random() < 0.5:
            pts = [(x1, y1), (x2, y1), (x2, y2)]
        else:
            pts = [(x1, y1), (x1, y2), (x2, y2)]
        d.line(pts, fill=wire, width=w, joint="curve")
        if rng.random() < 0.25:
            r = max(2, stroke * 1.6)
            cx, cy = pts[1]
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=wire)
    # mild scan / export artefacts
    if rng.random() < 0.3:
        page = page.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.8)))
    if rng.random() < 0.4:
        buf = io.BytesIO()
        page.save(buf, "JPEG", quality=rng.randint(55, 90))
        page = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return page, labels


_LIB = _TAX = None


def _one(job):
    """Page i with its own seed: pages are independent, so they can be made
    in parallel and an interrupted run only redoes the missing ones."""
    global _LIB, _TAX
    i, seed, out, split = job
    name = f"syn_{i:05d}"
    img = out / "images" / split / f"{name}.png"
    if img.exists():
        return {}
    if _LIB is None:
        _LIB, _TAX = ensure_library(), Taxonomy.load()
    page, labels = make_page(_LIB, random.Random(seed * 1_000_003 + i), _TAX)
    W, H = page.size
    counts, lines = {}, []
    for c, (x1, y1, x2, y2) in labels:
        lines.append(f"{c} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
        counts[_TAX.classes[c]] = counts.get(_TAX.classes[c], 0) + 1
    (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    page.save(img)                       # image last: it marks the page as done
    return counts


def main() -> None:
    import os
    from multiprocessing import Pool

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pages", type=int, default=1500)
    ap.add_argument("--val", type=float, default=0.08)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "yolo" / "synthetic")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    args = ap.parse_args()
    ensure_library()                      # unpack the symbols once, before the workers start
    for split in ("train", "val"):
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)
    n_val = int(args.pages * args.val)
    jobs = [(i, args.seed, args.out, "val" if i < n_val else "train") for i in range(args.pages)]
    counts: dict = {}
    with Pool(max(1, args.workers)) as pool:
        for k, c in enumerate(pool.imap_unordered(_one, jobs, chunksize=8), 1):
            for name, v in c.items():
                counts[name] = counts.get(name, 0) + v
            if k % 500 == 0:
                print(f"  {k}/{args.pages} pages", flush=True)
    print("synthetic parts per class (new pages):", dict(sorted(counts.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
