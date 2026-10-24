"""Read text in schematic crops.

Uses the PP-OCRv4 recogniser bundled with RapidOCR (PaddleOCR on ONNX, runs on
CPU), called directly so decoding can be restricted to schematic characters:
the model is Chinese+English, and on tiny digits it otherwise answers with
Chinese characters. Text boxes come from the text detector (or annotations).

- crops are cleaned of wire lines (a wire touching a label reads as "[" or "]")
- crops are scaled 3x with nearest-neighbour first (keeps tiny glyph edges sharp)
- NO white border: tested on PCBnet val, a 4 px border cut exact reads from
  71% to 42%; the recogniser wants text filling the line height
- vertical labels are read in both 90-degree rotations; the more confident wins
- `ascii_only` limits decoding to schematic characters. Off by default:
  JLC schematics contain real Chinese text, and on val it cost 4-5 points.
  Role-aware cleanup (textfix.py) handles stray characters instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REC_HEIGHT = 48
SCHEMATIC_CHARS = set(chr(c) for c in range(32, 127)) | set("Ωμ±°")


@dataclass
class Reading:
    text: str
    conf: float
    rotated: int     # 0, 90 or 270


def _model_path() -> Path:
    import rapidocr_onnxruntime

    return Path(rapidocr_onnxruntime.__file__).parent / "models" / "ch_PP-OCRv4_rec_infer.onnx"


class Reader:
    def __init__(self, threads: int = 4, ascii_only: bool = False):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(_model_path()), opts, providers=["CPUExecutionProvider"])
        chars = self.session.get_modelmeta().custom_metadata_map["character"].splitlines()
        # Output classes: 0 = CTC blank, then the dictionary, then space.
        self.classes = ["<blank>"] + chars + [" "]
        allowed = np.array([i == 0 or c in SCHEMATIC_CHARS for i, c in enumerate(self.classes)])
        self.blocked = ~allowed if ascii_only else np.zeros_like(allowed)
        self.input_name = self.session.get_inputs()[0].name

    def _rec(self, crop: np.ndarray) -> tuple[str, float]:
        # Same preprocessing as PaddleOCR / RapidOCR: BGR, height 48, padded
        # with zeros to at least 320 px wide.
        h, w = crop.shape[:2]
        new_w = int(np.clip(np.ceil(w * REC_HEIGHT / max(h, 1)), 1, REC_HEIGHT * 40))
        img = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), (new_w, REC_HEIGHT),
                         interpolation=cv2.INTER_CUBIC if h < REC_HEIGHT else cv2.INTER_AREA)
        x = np.zeros((3, REC_HEIGHT, max(new_w, 320)), np.float32)
        x[:, :, :new_w] = (img.astype(np.float32).transpose(2, 0, 1) / 255.0 - 0.5) / 0.5
        x = x[None]
        probs = self.session.run(None, {self.input_name: x})[0][0]      # (T, classes)
        probs[:, self.blocked] = 0
        idx = probs.argmax(1)
        best = probs.max(1)
        text, confs, prev = [], [], 0
        for i, p in zip(idx, best):
            if i != 0 and i != prev:
                text.append(self.classes[i])
                confs.append(p)
            prev = i
        if not text:
            return "", 0.0
        return "".join(text).strip(), float(np.mean(confs))

    def read(self, page: np.ndarray, box, line_mask: np.ndarray | None = None, pad: int = 2, scale: int = 3) -> Reading:
        """page: RGB array; line_mask: optional bool array of wire pixels to whiten."""
        x1, y1, x2, y2 = box
        H, W = page.shape[:2]
        a, b = max(int(x1) - pad, 0), max(int(y1) - pad, 0)
        c, d = min(int(np.ceil(x2)) + pad + 1, W), min(int(np.ceil(y2)) + pad + 1, H)
        crop = page[b:d, a:c].copy()
        if line_mask is not None:
            crop[line_mask[b:d, a:c]] = 255
        h, w = crop.shape[:2]
        if h == 0 or w == 0:
            return Reading("", 0.0, 0)
        if scale > 1:
            crop = cv2.resize(crop, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
        candidates = [(self._rec(crop), 0)]
        if h > w * 1.2:
            candidates += [(self._rec(cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)), 90),
                           (self._rec(cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)), 270)]
        (text, conf), rot = max(candidates, key=lambda t: t[0][1])
        return Reading(text, round(conf, 4), rot)


def wire_lines(rgb: np.ndarray, text_height: float) -> np.ndarray:
    """Long straight runs (wires, box outlines) to whiten inside text crops."""
    from .wires import ink_mask

    m = ink_mask(rgb).astype(np.uint8)
    run = int(max(10, 1.6 * text_height))
    lines = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((1, run), np.uint8)) | \
        cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((run, 1), np.uint8))
    return lines.astype(bool)
