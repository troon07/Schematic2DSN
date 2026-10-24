"""Text stage: text boxes (+ roles) -> recognised, cleaned Text objects."""

from __future__ import annotations

import numpy as np

from .ocr import Reader, wire_lines
from .pipeline import Text
from .textfix import clean

ROLE_TO_FIX = {"designator": "designator", "value": "value", "net_label": "netlabel",
               "pin_number": "pinnumber", "pin_name": "pinname"}


def read_texts(rgb: np.ndarray, boxes: list[tuple], roles: list[str], reader: Reader | None = None,
               role_confs: list[float] | None = None) -> list[Text]:
    """OCR each box, clean by role. conf = OCR conf x role conf."""
    reader = reader or Reader()
    heights = [min(b[2] - b[0], b[3] - b[1]) for b in boxes] or [14]
    lines = wire_lines(rgb, float(np.median(heights)))
    out = []
    for i, (box, role) in enumerate(zip(boxes, roles)):
        r = reader.read(rgb, box, lines)
        text = clean(r.text, ROLE_TO_FIX.get(role, ""), r.conf)
        conf = r.conf * (role_confs[i] if role_confs else 1.0)
        out.append(Text(text, tuple(box), role, round(conf, 3)))
    return out


_ENGINE = None


def ocr_page(rgb: np.ndarray, min_conf: float = 0.5) -> list[Text]:
    """Find and read all text on a page with RapidOCR's own detector (used
    when the trained text detector is not available). Roles come from the
    words (textroles.role_from_content)."""
    import re

    from rapidocr_onnxruntime import RapidOCR

    from .textroles import role_from_content

    global _ENGINE
    if _ENGINE is None:
        _ENGINE = RapidOCR()
    # use_cls off: the angle classifier flips some words ("Output" -> "indino O")
    res, _ = _ENGINE(np.ascontiguousarray(rgb[:, :, ::-1]), use_cls=False)
    out = []
    for box, text, conf in res or []:
        conf = float(conf)
        xs, ys = [p[0] for p in box], [p[1] for p in box]
        bb = (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)))
        t = text.strip()
        t = re.sub(r"^[Oo](?=[A-Z][a-z])", "", t)          # terminal circle read as "O": "OInput"
        role = role_from_content(t) if conf >= min_conf else "other"
        t = clean(t, ROLE_TO_FIX.get(role, ""), conf)
        out.append(Text(t, bb, role, round(conf, 3)))
    return out


def ocr_inside_parts(rgb: np.ndarray, parts, texts: list[Text], zoom: int = 2) -> list[Text]:
    """Second pass for parts with no designator nearby: designators drawn
    inside a symbol (Q8 between a transistor's leads) are often missed on the
    full page; OCR of the zoomed part box finds them."""
    from PIL import Image

    from .textroles import role_from_content

    if _ENGINE is None:
        ocr_page(rgb[:8, :8])                       # creates the engine
    from .textmatch import match_texts

    named = {m.part for m in match_texts(parts, texts) if m.kind == "designator"}
    H, W = rgb.shape[:2]
    out = []
    for p in parts:
        if p.id in named:
            continue
        x1, y1, x2, y2 = p.box
        size = max(x2 - x1, y2 - y1)
        m = int(0.5 * size) + 10                    # labels sit inside or right beside the symbol
        a, b_, c, d = max(int(x1) - m, 0), max(int(y1) - m, 0), min(int(x2) + m, W), min(int(y2) + m, H)
        if c - a < 8 or d - b_ < 8:
            continue
        crop = Image.fromarray(rgb[b_:d, a:c]).resize(((c - a) * zoom, (d - b_) * zoom), Image.LANCZOS)
        res, _ = _ENGINE(np.ascontiguousarray(np.asarray(crop)[:, :, ::-1]), use_cls=False)
        for box, text, conf in res or []:
            if role_from_content(text.strip()) != "designator" or float(conf) < 0.6:
                continue
            xs, ys = [q[0] / zoom + a for q in box], [q[1] / zoom + b_ for q in box]
            bb = (min(xs), min(ys), max(xs), max(ys))
            if any(min(bb[2], t.box[2]) > max(bb[0], t.box[0]) and min(bb[3], t.box[3]) > max(bb[1], t.box[1])
                   for t in texts + out):
                continue                            # already found
            out.append(Text(clean(text.strip(), "designator", float(conf)), bb, "designator", round(float(conf), 3)))
    return out
