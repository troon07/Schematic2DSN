"""Open Schematics (Hugging Face bshada/open-schematics, CC BY 4.0) -> YOLO.

~85k real schematics from public hardware projects. For KiCad 6+ files
(.kicad_sch) the dataset also has a PNG of the sheet, so exact component
boxes follow from the file itself: every placed symbol, its position,
rotation and mirror, and its graphics in the embedded lib_symbols.

    python scripts/prepare_open_schematics.py --shards 60            # first 60 shards
    python scripts/prepare_open_schematics.py --shards 300 --start 60  # continue later

Shards (~30 MB each, 939 in total) are downloaded one at a time, converted
and deleted; data/yolo/open_schematics/.shards/<n>.done marks finished ones,
so an interrupted run continues where it stopped. Pages go to train / val /
test by project name (all sheets of one project land in the same split).

Classes come from the reference prefix (R, C, Q, U, J, ...) refined by the
symbol name (LED, MOSFET, op-amp ...); prefixes that say nothing (project logos,
custom graphics) are skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from circuit2dsn.taxonomy import Taxonomy  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
URL = "https://huggingface.co/datasets/bshada/open-schematics/resolve/main/data/train-{:05d}.parquet"
N_SHARDS = 939
PAPER = {"A5": (210, 148), "A4": (297, 210), "A3": (420, 297), "A2": (594, 420), "A1": (841, 594),
         "A0": (1189, 841), "A": (279.4, 215.9), "B": (431.8, 279.4), "C": (558.8, 431.8),
         "D": (863.6, 558.8), "E": (1117.6, 863.6), "USLetter": (279.4, 215.9), "USLegal": (355.6, 215.9),
         "USLedger": (431.8, 279.4)}


# ---------------------------------------------------------------- s-expressions

def sexp(text: str):
    stack, cur, i, n = [], [], 0, len(text)
    while i < n:
        c = text[i]
        if c in " \t\r\n":
            i += 1
        elif c == "(":
            stack.append(cur)
            cur = []
            i += 1
        elif c == ")":
            done, cur = cur, stack.pop()
            cur.append(done)
            i += 1
        elif c == '"':
            j, buf = i + 1, []
            while text[j] != '"':
                if text[j] == "\\":
                    j += 1
                buf.append(text[j])
                j += 1
            cur.append(("S", "".join(buf)))
            i = j + 1
        else:
            j = i
            while j < n and text[j] not in ' \t\r\n()"':
                j += 1
            cur.append(text[i:j])
            i = j
    return cur[0] if cur else []


def kids(node, key):
    return [x for x in node if isinstance(x, list) and x and x[0] == key]


def kid(node, key):
    for x in node:
        if isinstance(x, list) and x and x[0] == key:
            return x
    return None


def s(v):
    return v[1] if isinstance(v, tuple) else v


# ---------------------------------------------------------------- geometry

def local_points(sym, unit: int, style: int) -> list[tuple[float, float]]:
    """Outline points of one unit (plus the shared unit 0) of an embedded lib
    symbol, KiCad library coordinates (mm, y up). Hidden pins and text skipped."""
    pts = []
    for sub in kids(sym, "symbol"):
        m = re.search(r"_(\d+)_(\d+)$", s(sub[1]))
        if not m or int(m.group(1)) not in (0, unit) or int(m.group(2)) not in (0, style):
            continue
        for el in sub:
            if not isinstance(el, list) or not el:
                continue
            k = el[0]
            if k == "polyline":
                pts += [(float(p[1]), float(p[2])) for p in kids(kid(el, "pts") or [], "xy")]
            elif k == "rectangle":
                for key in ("start", "end"):
                    p = kid(el, key)
                    pts.append((float(p[1]), float(p[2])))
            elif k == "circle":
                c, r = kid(el, "center"), float(kid(el, "radius")[1])
                pts += [(float(c[1]) - r, float(c[2]) - r), (float(c[1]) + r, float(c[2]) + r)]
            elif k == "arc":
                for key in ("start", "mid", "end"):
                    p = kid(el, key)
                    if p:
                        pts.append((float(p[1]), float(p[2])))
            elif k == "bezier":
                pts += [(float(p[1]), float(p[2])) for p in kids(kid(el, "pts") or [], "xy")]
            elif k == "pin":
                if "hide" in el or kid(el, "hide") is not None and s(kid(el, "hide")[1]) != "no":
                    continue
                at, ln = kid(el, "at"), kid(el, "length")
                x, y = float(at[1]), float(at[2])
                a = math.radians(float(at[3]) if len(at) > 3 else 0.0)
                L = float(ln[1]) if ln else 0.0
                pts += [(x, y), (x + L * math.cos(a), y + L * math.sin(a))]
    return pts


def place(pts, at, mirror: str | None):
    """Library points -> sheet coordinates (mm, y down)."""
    X, Y = float(at[1]), float(at[2])
    ang = int(round(float(at[3]) if len(at) > 3 else 0.0)) % 360
    out = []
    for lx, ly in pts:
        x, y = lx, -ly                       # library y up -> sheet y down
        if mirror == "x":
            y = -y
        elif mirror == "y":
            x = -x
        for _ in range(ang // 90):           # counter-clockwise on screen
            x, y = y, -x
        out.append((X + x, Y + y))
    return out


# ---------------------------------------------------------------- classes

def class_of(ref: str, lib_id: str, value: str) -> str | None:
    ref_u, name = ref.upper(), (lib_id + " " + value).upper()
    if ref_u.startswith("#PWR") or lib_id.startswith("power:"):
        return "ground" if re.search(r"GND|EARTH|VSS|0V|CHASSIS", name) else "power"
    if ref_u.startswith("#FLG") or ref_u.startswith("#"):
        return None                                     # PWR_FLAG etc.: not drawn on paper
    m = re.match(r"([A-Z]+)", ref_u)
    prefix = m.group(1) if m else ""
    if prefix == "LED" or re.search(r"\bLED|LED_|_LED|NEOPIX|WS28", name):
        return "led"
    if re.search(r"TESTPOINT|TEST_POINT", name):
        return "terminal"
    if prefix in ("IMG", "LOGO", "GB", "GF", "SYM", "GRAPHIC", "FID", "G", "REF"):
        return None                                     # logos, fiducials, custom graphics: not parts
    if prefix in ("JP", "SJ", "J") and re.search(r"JUMPER|SOLDERJUMPER|BRIDGE", name):
        return "jumper"
    if prefix in ("R", "RS", "RN", "RA"):
        return "potentiometer" if re.search(r"POT|TRIM", name) else "resistor"
    if prefix in ("RV", "VR", "POT"):
        return "varistor" if re.search(r"VARISTOR|MOV", name) else "potentiometer"
    if prefix in ("C", "CP"):
        return "capacitor"
    if prefix in ("L", "FB"):
        return "inductor"
    if prefix in ("D", "CR", "ZD", "TVS"):
        return "diode"
    if prefix in ("Q", "T", "M") and re.search(r"FET|MOS|NMOS|PMOS|2N7002|BSS|IRF|AO3|SI23|DMG", name):
        return "mosfet"
    if prefix in ("Q",):
        return "thyristor" if re.search(r"SCR|THYRISTOR", name) else "triac" if "TRIAC" in name else "bjt"
    if prefix in ("T", "TR") and re.search(r"TRANSFORMER|XFMR|TRAFO", name):
        return "transformer"
    if prefix in ("U", "IC", "A", "AR", "OA") and re.search(r"OPAMP|OP-AMP|OPA\d|LM358|LM324|TL07|TL08|MCP60|AD86|LMV3|NE5532|OPAMP", name):
        return "opamp"
    if prefix in ("U", "IC", "A", "J", "P", "CN", "CON", "X", "XS", "XP", "MOD", "M", "REG", "VREG", "LDO", "PS", "DC",
                  "CONN", "HDR", "USB", "MCU", "SOC") :
        if prefix == "M" and re.search(r"MOTOR", name):
            return "motor"
        if prefix == "X" and re.search(r"CRYSTAL|XTAL|OSC|MHZ|KHZ", name):
            return "crystal"
        return "ic_box"
    if prefix in ("Y", "XTAL", "Q") or re.search(r"CRYSTAL|XTAL", name):
        return "crystal"
    if prefix in ("F", "FUSE", "PTC"):
        return "fuse"
    if prefix in ("SW", "S", "BTN", "K") and not (prefix == "K" and re.search(r"RELAY", name)):
        return "relay" if re.search(r"RELAY", name) else "switch"
    if prefix in ("K", "RL", "RLY"):
        return "relay"
    if prefix in ("BT", "BAT", "B"):
        return "battery"
    if prefix in ("BZ", "BUZ"):
        return "buzzer"
    if prefix in ("LS", "SP", "SPK"):
        return "speaker"
    if prefix in ("MK", "MIC"):
        return "microphone"
    if prefix in ("AE", "ANT"):
        return "antenna"
    if prefix in ("JP", "SJ"):
        return "jumper"
    if prefix in ("H", "MH", "HOLE"):
        return "mounting_hole"
    if prefix in ("TP",):
        return "terminal"
    if prefix in ("NT",):
        return "net_tie"
    if prefix in ("OK", "ISO", "OC") or re.search(r"OPTO|PC817|4N2|TLP", name):
        return "optocoupler"
    if prefix in ("LA", "LAMP"):
        return "lamp"
    # unknown prefix: trust the symbol name if it says what it is, else skip
    for key, cls in (("RESISTOR", "resistor"), ("CAPACITOR", "capacitor"), ("DIODE", "diode"),
                     ("TRANSISTOR", "bjt"), ("INDUCTOR", "inductor"), ("CONN", "ic_box")):
        if key in name:
            return cls
    return None


# ---------------------------------------------------------------- one sheet

def convert(sch_text: str, img_w: int, img_h: int, tax: Taxonomy):
    tree = sexp(sch_text)
    if not tree or tree[0] != "kicad_sch":
        return None
    paper = kid(tree, "paper")
    if paper is None:
        return None
    pname = s(paper[1])
    if pname == "User":
        pw, ph = float(paper[2]), float(paper[3])
    elif pname in PAPER:
        pw, ph = PAPER[pname]
    else:
        return None
    if "portrait" in paper:
        pw, ph = ph, pw
    sx, sy = img_w / pw, img_h / ph
    if abs(sx - sy) / sx > 0.03:          # image is not the whole sheet
        return None
    libs = {s(sym[1]): sym for sym in kids(kid(tree, "lib_symbols") or [], "symbol")}
    labels = []
    for sym in kids(tree, "symbol"):
        lib_id = s((kid(sym, "lib_id") or [None, ""])[1])
        lib_key = s((kid(sym, "lib_name") or [None, lib_id])[1])     # a local copy wins
        at = kid(sym, "at")
        if at is None or lib_key not in libs:
            continue
        props = {s(p[1]): s(p[2]) for p in kids(sym, "property") if len(p) > 2}
        cls = class_of(props.get("Reference", ""), lib_id, props.get("Value", ""))
        if cls is None:
            continue
        unit = int((kid(sym, "unit") or [None, 1])[1])
        style = int((kid(sym, "convert") or kid(sym, "body_style") or [None, 1])[1])
        pts = local_points(libs[lib_key], unit, style)
        if len(pts) < 2:
            continue
        mir = kid(sym, "mirror")
        q = place(pts, at, s(mir[1]) if mir else None)
        x1, y1 = min(p[0] for p in q) * sx, min(p[1] for p in q) * sy
        x2, y2 = max(p[0] for p in q) * sx, max(p[1] for p in q) * sy
        pad = 1.0
        x1, y1, x2, y2 = max(x1 - pad, 0), max(y1 - pad, 0), min(x2 + pad, img_w), min(y2 + pad, img_h)
        if x2 - x1 < 3 or y2 - y1 < 3:
            continue
        labels.append((tax.index(cls), (x1 + x2) / 2 / img_w, (y1 + y2) / 2 / img_h, (x2 - x1) / img_w, (y2 - y1) / img_h))
    outside = 0
    for nc in kids(tree, "no_connect"):
        at = kid(nc, "at")
        cx, cy, r = float(at[1]) * sx, float(at[2]) * sy, 0.9 * sx
        if not (r <= cx <= img_w - r and r <= cy <= img_h - r):
            outside += 1
            continue
        labels.append((tax.index("no_connect"), cx / img_w, cy / img_h, 2 * r / img_w, 2 * r / img_h))
    # every box inside the page; a page whose drawing does not match its file
    # (many parts outside the picture) is not used at all
    clean = []
    for c, x, y, w, h in labels:
        x1, y1, x2, y2 = max(x - w / 2, 0.0), max(y - h / 2, 0.0), min(x + w / 2, 1.0), min(y + h / 2, 1.0)
        if x2 - x1 <= 0 or y2 - y1 <= 0 or (x2 - x1) * (y2 - y1) < 0.6 * w * h:
            outside += 1
            continue
        clean.append((c, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
    if outside > 0.1 * max(len(labels), 1) + 1:
        return None
    # the same part listed twice (multi-unit symbols drawn on top of each other)
    seen, uniq = set(), []
    for lab in clean:
        key = (lab[0],) + tuple(round(v, 4) for v in lab[1:])
        if key not in seen:
            seen.add(key)
            uniq.append(lab)
    return uniq


def split_of(name: str, val: float, test: float) -> str:
    h = int(hashlib.md5(name.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if h < test else "val" if h < test + val else "train"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", type=int, default=60, help="how many shards to process (from --start)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "yolo" / "open_schematics")
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "raw" / "open_schematics")
    ap.add_argument("--min-parts", type=int, default=3, help="skip sheets with fewer components (title pages)")
    ap.add_argument("--debug", type=int, default=6, help="draw boxes on this many pages for a visual check")
    args = ap.parse_args()
    import pyarrow.parquet as pq
    from PIL import Image, ImageDraw

    import socket

    socket.setdefaulttimeout(60)      # a dead connection fails (and is retried) instead of hanging forever
    tax = Taxonomy.load()
    done_dir = args.out / ".shards"
    done_dir.mkdir(parents=True, exist_ok=True)
    args.raw.mkdir(parents=True, exist_ok=True)
    for sp in ("train", "val", "test"):
        (args.out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / sp).mkdir(parents=True, exist_ok=True)
    stats = json.loads((args.out / "stats.json").read_text()) if (args.out / "stats.json").exists() else {}
    debug_left = args.debug
    for shard in range(args.start, min(args.start + args.shards, N_SHARDS)):
        marker = done_dir / f"{shard}.done"
        if marker.exists():
            continue
        path = args.raw / f"train-{shard:05d}.parquet"
        t0 = time.time()
        for attempt in range(5):
            try:
                if not path.exists():
                    tmp = path.with_suffix(".part")
                    urllib.request.urlretrieve(URL.format(shard), tmp)
                    tmp.rename(path)
                table = pq.read_table(path, columns=["schematic", "schematic_image", "name"])
                break
            except Exception as e:                       # noqa: BLE001 - network / partial file: retry
                print(f"shard {shard}: {type(e).__name__} {e}; retry {attempt + 1}", flush=True)
                path.unlink(missing_ok=True)
                time.sleep(5 * (attempt + 1))
                # internet gone (name lookup fails): wait for it, up to 15 min, instead of using up retries
                for _ in range(90):
                    try:
                        socket.gethostbyname("huggingface.co")
                        break
                    except OSError:
                        time.sleep(10)
        else:
            continue
        pages = boxes = 0
        for k, row in enumerate(table.to_pylist()):
            sch, img = row.get("schematic"), row.get("schematic_image")
            if not sch or not img or "(kicad_sch" not in sch[:300]:
                continue
            try:
                im = Image.open(io.BytesIO(img["bytes"] if isinstance(img, dict) else img)).convert("RGB")
                labels = convert(sch, im.width, im.height, tax)
            except Exception:                            # noqa: BLE001 - malformed file: skip it
                continue
            if not labels or len(labels) < args.min_parts:
                continue
            sp = split_of(row.get("name") or f"{shard}_{k}", 0.04, 0.04)
            stem = f"os{shard:03d}_{k:03d}"
            im.save(args.out / "images" / sp / f"{stem}.jpg", quality=92)
            (args.out / "labels" / sp / f"{stem}.txt").write_text(
                "".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in labels))
            for c, *_ in labels:
                stats[tax.classes[c]] = stats.get(tax.classes[c], 0) + 1
            pages += 1
            boxes += len(labels)
            if debug_left > 0:
                d = ImageDraw.Draw(im)
                for c, x, y, w, h in labels:
                    d.rectangle([(x - w / 2) * im.width, (y - h / 2) * im.height, (x + w / 2) * im.width,
                                 (y + h / 2) * im.height], outline=(255, 0, 0))
                    d.text(((x - w / 2) * im.width, (y - h / 2) * im.height - 10), tax.classes[c][:6], fill=(255, 0, 0))
                im.save(args.out / f"debug_{stem}.png")
                debug_left -= 1
        path.unlink(missing_ok=True)
        marker.write_text(json.dumps({"pages": pages, "boxes": boxes}))
        stats["_pages"] = stats.get("_pages", 0) + pages
        (args.out / "stats.json").write_text(json.dumps(stats, indent=1))
        print(f"shard {shard}: {pages} pages, {boxes} boxes ({time.time() - t0:.0f}s); total pages {stats['_pages']}",
              flush=True)
    names = Taxonomy.load().classes
    (args.out / "dataset.yaml").write_text(
        f"path: {args.out.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)))


if __name__ == "__main__":
    main()
