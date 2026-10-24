"""Second opinion on detected parts from a vision-language model (VLM).

The detector finds and classifies every part. For each part it is not sure of
(and every IC), a small crop is shown to a VLM that is told the allowed classes
and asked to read what is printed on it (part number such as 74F244, reference
such as U10). The answer is merged back:

  * agrees with the detector      -> confidence goes up
  * disagrees                     -> the part is flagged for a person (never silently changed)
  * part number is in the OrCAD   -> an OrCAD suggestion is attached (library, part, pins)
    catalogue
  * part number is not in OrCAD   -> marked "unidentified" with the text that was read

Backends: "hf" (transformers, 4-bit, for Colab) and "ollama" (local HTTP).
Default model: Qwen2.5-VL-7B-Instruct, the strongest open model that fits a free 16 GB T4
in 4-bit and reads small printed text well.
"""

from __future__ import annotations

import base64
import io
import json
import re
from collections import Counter
from pathlib import Path

from PIL import Image

HF_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
OLLAMA_MODEL = "qwen2.5vl:7b"

PROMPT = """You are looking at one cropped part from an electronic circuit schematic.
Allowed classes: {classes}.
Answer with JSON only, no other text:
{{"cls": "<one allowed class, or unknown>", "part_number": "<printed chip/part number such as 74F244, or empty>",
 "ref": "<reference such as U10, R5, or empty>", "value": "<value such as 1K00, or empty>", "reason": "<few words>"}}
Rules: a triangle (with or without a small circle) is buffer_gate or not_gate; a rectangle with pin numbers and a
chip number is ic_box; read text exactly as printed; if unsure use "unknown"."""


class HFBackend:
    def __init__(self, model: str = HF_MODEL, four_bit: bool = True):
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                               bnb_4bit_quant_type="nf4") if four_bit else None
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model, torch_dtype=torch.float16, device_map="auto", quantization_config=q)
        self.proc = AutoProcessor.from_pretrained(model, min_pixels=256 * 28 * 28, max_pixels=640 * 28 * 28)

    def ask(self, image: Image.Image, prompt: str) -> str:
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.proc.apply_chat_template(msgs, add_generation_prompt=True)
        inp = self.proc(text=[text], images=[image], return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inp, max_new_tokens=120, do_sample=False)
        return self.proc.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0]


class OllamaBackend:
    def __init__(self, model: str = OLLAMA_MODEL, url: str = "http://localhost:11434"):
        self.model, self.url = model, url.rstrip("/")

    def ask(self, image: Image.Image, prompt: str) -> str:
        import urllib.request

        buf = io.BytesIO()
        image.save(buf, "PNG")
        body = json.dumps({"model": self.model, "stream": False, "options": {"temperature": 0},
                           "messages": [{"role": "user", "content": prompt,
                                         "images": [base64.b64encode(buf.getvalue()).decode()]}]}).encode()
        req = urllib.request.Request(self.url + "/api/chat", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read())["message"]["content"]


def parse(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return d if isinstance(d, dict) else None


def crop(img: Image.Image, c: dict, pad: float = 0.35, min_side: int = 96) -> Image.Image:
    w, h = c["x2"] - c["x1"], c["y2"] - c["y1"]
    px, py = max(w * pad, 12), max(h * pad, 12)
    box = (max(c["x1"] - px, 0), max(c["y1"] - py, 0), min(c["x2"] + px, img.width), min(c["y2"] + py, img.height))
    cr = img.crop(tuple(int(v) for v in box)).convert("RGB")
    s = max(1.0, min_side / max(cr.size), 448 / max(cr.size))
    return cr.resize((int(cr.width * s), int(cr.height * s)), Image.LANCZOS)


def _orcad_match(part_number: str, want_pins: int | None = None):
    """Exact catalogue match (also with the 'A' revision suffix). When several libraries have the part,
    take the one whose pin count is closest to what was detected on the drawing."""
    try:
        from . import orcad_parts
    except Exception:
        return None
    if not part_number or not orcad_parts.available():
        return None
    pn = re.sub(r"[^A-Za-z0-9]", "", part_number).lower()
    if len(pn) < 3:
        return None
    hits = []
    for r in orcad_parts.search(pn, limit=40):
        key = re.sub(r"[^a-z0-9]", "", r["name"].lower())
        if key == pn or key == pn + "a":
            hits.append(r)
    if not hits:
        return None
    if want_pins:
        hits.sort(key=lambda r: (abs(r["pins"] - want_pins), key_len(r["name"])))
    r = hits[0]
    return {"file": r["file"], "lib": r["lib"], "name": r["name"], "pins": r["pins"]}


def key_len(name: str) -> int:
    return len(name)


REF_LIKE = re.compile(r"^[A-Za-z]{1,2}\d{1,3}$")


def _guesses(pn: str, limit: int = 3) -> list[dict]:
    """Close spellings in the OrCAD catalogue for a number that is not there as read
    (the model may have misread a character). Suggestions only: never applied by themselves."""
    import difflib

    try:
        from . import orcad_parts
    except Exception:
        return []
    key = re.sub(r"[^A-Za-z0-9]", "", pn).lower()
    if len(key) < 4 or not orcad_parts.available():
        return []
    pool = {}
    for r in orcad_parts.index():
        if r["_key"][:2] == key[:2]:
            pool.setdefault(r["_key"], r)
    def score(k: str) -> float:                    # similarity, ties broken by the longest shared start
        pre = next((i for i, (x, y) in enumerate(zip(key, k)) if x != y), min(len(key), len(k)))
        return difflib.SequenceMatcher(None, key, k).ratio() + 0.02 * pre

    out = []
    for k in sorted((k for k in pool if difflib.SequenceMatcher(None, key, k).ratio() >= 0.8), key=score, reverse=True)[:limit]:
        r = pool[k]
        out.append({"file": r["file"], "lib": r["lib"], "name": r["name"], "pins": r["pins"]})
    return out


def enrich(v: dict, det_cls: str, want_pins: int | None = None) -> dict:
    """Clean what the model read and look it up in OrCAD. Recomputed every time, so old result files
    benefit from later improvements."""
    pn = str(v.get("part_number", "")).strip()
    if pn and REF_LIKE.match(pn):                  # "U42" is a reference designator, not a part number
        v["ref"] = v.get("ref") or pn
        pn = ""
    v["part_number"] = pn
    for k in ("orcad", "unidentified", "guess"):
        v.pop(k, None)
    om = _orcad_match(pn, want_pins)
    if om:
        v["orcad"] = om
    elif pn and det_cls == "ic_box":
        v["unidentified"] = True                   # printed number not in the OrCAD libraries
        g = _guesses(pn)
        if g:
            v["guess"] = g
    return v


LOGIC = {"and_gate", "or_gate", "nand_gate", "nor_gate", "xor_gate", "xnor_gate", "not_gate", "buffer_gate",
         "opamp", "ic_box"}


def judge(det_cls: str, det_conf: float, v: dict) -> tuple[float, str | None, bool | None]:
    """Compare the detector with the vision model. Returns (confidence, flag, agrees).
    agrees is None when the vision model could not tell ("unknown"): that says nothing either way.
    A chip whose printed number is an OrCAD part is a good IC whatever its drawing looks like
    (a 74F244 buffer is drawn as a triangle), so ic_box vs a logic symbol still counts as agreement."""
    cls = v.get("cls", "unknown")
    if cls in ("", "unknown"):
        return det_conf, None, None
    if cls == det_cls:
        conf, agrees = det_conf + (1 - det_conf) * 0.5, True
    elif det_cls == "ic_box" and cls in LOGIC and v.get("orcad"):
        conf, agrees = det_conf + (1 - det_conf) * 0.5, True
    else:
        return min(det_conf, 0.60), f"detector says {det_cls}, vision model says {cls}", False
    if v.get("orcad"):
        conf = max(conf, 0.85)
    return round(min(conf, 0.99), 4), None, agrees


def verify(img: Image.Image, comps: list[dict], backend, classes: list[str], below: float = 0.80,
           always: tuple[str, ...] = ("ic_box",), progress=None) -> list[dict]:
    """Adds c["vlm"] = {...} to the checked parts and updates conf/band/flag in place."""
    prompt = PROMPT.format(classes=", ".join(classes))
    todo = [c for c in comps if c["conf"] < below or c["cls"] in always]
    for i, c in enumerate(todo):
        ans = parse(backend.ask(crop(img, c), prompt))
        if progress:
            progress(i + 1, len(todo))
        if not ans:
            continue
        cls = str(ans.get("cls", "unknown")).strip()
        pn = str(ans.get("part_number", "")).strip()
        v = {"cls": cls, "part_number": pn, "ref": str(ans.get("ref", "")).strip(),
             "value": str(ans.get("value", "")).strip(), "reason": str(ans.get("reason", ""))[:80]}
        enrich(v, c["cls"])
        conf, flag, agrees = judge(c["cls"], c["conf"], v)
        v["agrees"] = agrees
        c["vlm"] = v
        c["conf"] = conf
        if flag:
            c["flag"] = flag
        c["band"] = "high" if conf >= 0.8 else "medium" if conf >= 0.5 else "low"
    return comps


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def apply_vlm(parts: list[dict], vlm_components: list[dict], min_iou: float = 0.6) -> int:
    """Merge a *.vlm.json (from scripts/vlm_verify.py) into a netlist's parts, matching by
    box overlap. Sets part["vlm"] and the vision-checked confidence. Returns parts updated."""
    boxes = [(c, (c["x1"], c["y1"], c["x2"], c["y2"])) for c in vlm_components if "vlm" in c]
    n = 0
    for p in parts:
        best = max(((_iou(p["box"], b), c) for c, b in boxes), key=lambda t: t[0], default=(0, None))
        if best[0] < min_iou:
            continue
        c = best[1]
        v = enrich(dict(c["vlm"]), p["cls"], len(p.get("pins") or []) or None)
        conf, flag, agrees = judge(p["cls"], p["conf"], v)    # judged against THIS detection, not the stored one
        v["agrees"] = agrees
        p["vlm"] = v
        p["conf"] = conf
        if flag:
            p["vlm_flag"] = flag
        if (agrees and p["cls"] == "ic_box" and v.get("orcad") and not p.get("orcad") and conf >= 0.85):
            p["orcad"] = _orcad_dict(v["orcad"])            # drawn with OrCAD's own symbol, pin names and numbers
        n += 1
    snap_numbers(parts)
    return n


def add_ocr(img: Image.Image, parts: list[dict], progress=None) -> int:
    """Run the independent chip-number reader on every IC. Stored as p["vlm"]["ocr"] (a "vlm" entry is
    created when the vision model has not looked at the part, so the pipeline works with OCR alone)."""
    from . import chip_ocr

    ics = [p for p in parts if p["cls"] == "ic_box"]
    for k, p in enumerate(ics):
        nums = chip_ocr.read_numbers(img, p["box"])[:3]
        v = p.setdefault("vlm", {"cls": "unknown", "part_number": "", "ref": "", "value": "", "reason": "",
                                 "agrees": None, "source": "ocr"})
        v["ocr"] = nums
        if not v.get("part_number") and nums:
            v["part_number"] = nums[0]
        if progress:
            progress(k + 1, len(ics))
    return len(ics)


def _norm(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", s).lower()


def _dist(a: str, b: str) -> int:
    import difflib

    return max(len(a), len(b)) - sum(m.size for m in difflib.SequenceMatcher(None, a, b).get_matching_blocks())


def snap_numbers(parts: list[dict]) -> int:
    """Combine the two readers and the chip types seen elsewhere on the sheet.
      * both readers give the same OrCAD part            -> trusted (90%)
      * a reading is a close spelling (<= 2 characters) of an OrCAD part that was read cleanly
        elsewhere on the sheet and no reading disagrees much -> that part, but a person confirms (68%)
    Misreads like 74F24 or 74P163 are fixed by the many correct 74F244 / 74F163 on the same page."""
    vocab = Counter()
    for p in parts:
        v = p.get("vlm") or {}
        if v.get("orcad"):
            vocab[v["part_number"]] += 1
    n = 0
    for p in parts:
        v = p.get("vlm")
        if not v or p["cls"] != "ic_box":
            continue
        ocr = v.get("ocr") or []
        reads = [r for r in [v.get("part_number", "")] + ocr[:2] if r]
        want = len(p.get("pins") or []) or None
        if v.get("orcad"):                                         # vision model matched: does OCR agree?
            if any(_norm(r) == _norm(v["part_number"]) for r in ocr):
                p["conf"] = max(p["conf"], 0.9)
                v["agrees_ocr"] = True
            continue
        om = next((m for m in (_orcad_match(r, want) for r in ocr) if m), None)     # OCR alone found a real part
        best = None
        if om is None and reads:
            for cand, cnt in vocab.items():
                ds = [_dist(_norm(r), _norm(cand)) for r in reads]
                if min(ds) <= 2 and min(ds) <= 1 + (len(ds) > 1 and 1 or 0) and _norm(cand)[:3] in {_norm(r)[:3] for r in reads}:
                    score = (-sum(ds), cnt)
                    if best is None or score > best[0]:
                        best = (score, cand)
            if best:
                om = _orcad_match(best[1], want)
        if om is None:
            continue
        v["snapped_from"] = v.get("part_number") or ""
        v["part_number"], v["orcad"] = om["name"], om
        v.pop("unidentified", None)
        v.pop("guess", None)
        p["conf"] = min(max(p["conf"], 0.0), 0.68)
        p["vlm_flag"] = f"read as {', '.join(reads) or '?'} → matched {om['name']}: confirm"
        p["orcad"] = _orcad_dict(om)
        n += 1
    return n


def find_vlm_file(source: Path, page_no: int, extra_dirs=()) -> Path | None:
    name = f"{source.stem}_p{page_no}.vlm.json"
    for d in (source.parent, *extra_dirs):
        f = Path(d) / name
        if f.exists():
            return f
    return None


def _orcad_dict(om: dict) -> dict | None:
    from . import orcad_parts

    row = next((r for r in orcad_parts.index() if r["file"] == om["file"] and r["name"] == om["name"]), None)
    if row is None:
        return None
    return {"file": row["file"], "name": row["name"], "lib": row["lib"], "prefix": row["prefix"],
            "fp": row["fp"], "pins": row["pins"], "pin_map": {}}


UNKNOWN_PROMPT = """This crop from an electronic schematic shows drawing the part detector could not name.
Allowed classes: {classes}.
Answer JSON only: {{"cls": "<allowed class or unknown>", "part_number": "<printed number or empty>", "ref": "<reference or empty>"}}"""

JUNCTION_PROMPT = """The centre of this crop of a schematic is where wires meet or cross.
Answer JSON only: {"connected": true|false, "kind": "dot|cross|hop|corner"}
A solid dot where lines cross means connected. Lines crossing with no dot, or with a small arc (hop), are NOT connected."""


def check_netlist(img: Image.Image, netlist: dict, backend, classes: list[str], max_calls: int = 120,
                  progress=None) -> dict:
    """The automatic second pass after the detector and wire tracing: parts, unrecognised drawing and
    doubtful junctions. Results are hints on the netlist; a person still sees anything left under 70%."""
    stats = {"parts": 0, "unknowns": 0, "junctions": 0}
    comps = [{"id": p["id"], "cls": p["cls"], "conf": p["conf"], "x1": p["box"][0], "y1": p["box"][1],
              "x2": p["box"][2], "y2": p["box"][3]} for p in netlist["parts"] if "vlm" not in p]
    left = max_calls
    todo = [c for c in comps if c["conf"] < 0.8 or c["cls"] == "ic_box"][:left]
    comps_checked = verify(img, todo, backend, classes, below=1.1, always=(), progress=progress)
    stats["parts"] = apply_vlm(netlist["parts"], comps_checked)
    left -= len(todo)
    for u in netlist.get("unidentified", []):
        if left <= 0:
            break
        left -= 1
        b = u["box"]
        ans = parse(backend.ask(crop(img, {"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3]}),
                                UNKNOWN_PROMPT.format(classes=", ".join(classes))))
        if ans and ans.get("cls") not in (None, "", "unknown"):
            u["vlm"] = {"cls": ans["cls"], "part_number": str(ans.get("part_number", "")).strip()}
            u["why"] += f" · vision model thinks: {ans['cls']}" + (f" {u['vlm']['part_number']}" if u["vlm"]["part_number"] else "")
            stats["unknowns"] += 1
    spots = []
    for n in netlist["nets"]:
        for d in n.get("doubts", []):
            m = re.match(r"(dot|hop) at \((\d+),(\d+)\)", d.get("what", ""))
            if m:
                spots.append((d.get("conf", 1.0), n, d, m.group(1), float(m.group(2)), float(m.group(3))))
    spots.sort(key=lambda s: s[0])
    for conf, n, d, kind, x, y in spots[:max(left, 0)]:
        ans = parse(backend.ask(crop(img, {"x1": x - 20, "y1": y - 20, "x2": x + 20, "y2": y + 20}, pad=0.0, min_side=224),
                                JUNCTION_PROMPT))
        if not ans or "connected" not in ans:
            continue
        ours = kind == "dot"                                # we decided: dot = joined, hop = not joined
        d["vlm"] = bool(ans["connected"])
        stats["junctions"] += 1
        if bool(ans["connected"]) != ours:
            d["what"] += " · vision model disagrees"
            n["conf"] = min(n.get("conf", 1.0), 0.5)
            n["doubts"].sort(key=lambda q: "disagrees" not in q.get("what", ""))   # shown first in the check list
    for n in netlist["nets"]:                               # every checked junction of the net agreed: more trust
        ds = [d for d in n.get("doubts", []) if "vlm" in d]
        if ds and len(ds) == len(n["doubts"]) and not any("disagrees" in d["what"] for d in ds):
            n["conf"] = round(max(n.get("conf", 0), min(0.85, n.get("conf", 0) + 0.1)), 3)
    return stats
