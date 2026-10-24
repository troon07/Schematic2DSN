"""Second reader for printed chip numbers (74F244, 74LS298, TC514100 ...), independent of the vision model.

Chip numbers are often printed sideways inside the box. The crop is OCR'd upright and rotated both ways and
the tokens that look like a part number are kept. The review logic then trusts a number that BOTH readers
produced, and uses this one when the vision model returned nothing or something that is not a real part.
"""

from __future__ import annotations

import re

import numpy as np
from PIL import Image

TOKEN = re.compile(r"[A-Z0-9][A-Z0-9/\-]{3,14}")
_ENGINE = None


def looks_like_part(tok: str) -> bool:
    digits = sum(c.isdigit() for c in tok)
    return digits >= 2 and len(tok) >= 4 and not re.fullmatch(r"[A-Z]{1,2}\d{1,3}", tok) \
        and not re.fullmatch(r"\d{1,3}", tok) and not tok.startswith(("BB", "DD", "NW", "AA", "CC", "KK"))


def read_numbers(img: Image.Image, box, pad: float = 0.1) -> list[str]:
    """Candidate part numbers printed on/in the box, most plausible first."""
    global _ENGINE
    from rapidocr_onnxruntime import RapidOCR

    if _ENGINE is None:
        _ENGINE = RapidOCR()
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    c = img.convert("RGB").crop((max(int(x1 - w * pad), 0), max(int(y1 - h * pad), 0),
                                 min(int(x2 + w * pad), img.width), min(int(y2 + h * pad), img.height)))
    s = max(1.0, 200 / max(min(c.size), 1))
    c = c.resize((int(c.width * s), int(c.height * s)), Image.LANCZOS)
    found: dict[str, float] = {}
    for rot in (0, 90, 270):
        im = c.rotate(rot, expand=True) if rot else c
        res, _ = _ENGINE(np.ascontiguousarray(np.asarray(im)[:, :, ::-1]), use_cls=False)
        for _, text, conf in res or []:
            t = text.upper().replace(" ", "")
            for m in TOKEN.findall(t):
                if looks_like_part(m):
                    found[m] = max(found.get(m, 0.0), float(conf))
    return [t for t, _ in sorted(found.items(), key=lambda kv: -kv[1])]
