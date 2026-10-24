"""One command: sheet image/PDF -> parts -> wires -> second opinion (local vision model + chip OCR) -> OrCAD .dsn.

    python scripts/run_sheet.py colab_upload/9.jpg
    python scripts/run_sheet.py my_circuit.pdf --page 2 --vlm ollama      # live local vision model (ollama pull qwen2.5vl:7b)
    python scripts/run_sheet.py my_circuit.pdf --vlm hf                   # on a CUDA GPU / Colab

A saved vision-model result (inputs/vlm/<name>_p<N>.vlm.json, from Colab) is used when present.
Writes outputs/run/<name>_p<N>/ : .DSN, .OLB, PDF + PNG preview, netlist json, summary.txt
Anything under 70% sure is listed in summary.txt: open the dashboard (scripts/review.py) to check those.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from circuit2dsn import vlm_check  # noqa: E402
from circuit2dsn.detect import Detector, load_pages  # noqa: E402
from circuit2dsn.dsn_export import make_dsn  # noqa: E402
from circuit2dsn.pipeline import Part, extract_netlist  # noqa: E402
from circuit2dsn.review_state import VERIFY_BELOW  # noqa: E402
from circuit2dsn.taxonomy import Taxonomy  # noqa: E402
from circuit2dsn.textstage import ocr_inside_parts, ocr_page  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path)
    ap.add_argument("--page", type=int, default=1)
    ap.add_argument("--weights", type=Path, default=ROOT / "runs" / "detector" / "weights" / "best.pt")
    ap.add_argument("--vlm", choices=["none", "ollama", "hf"], default="none")
    ap.add_argument("--vlm-model")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "run")
    a = ap.parse_args()

    page = load_pages(a.input, a.dpi)[a.page - 1]
    parts = [Part(x.id, x.cls, (x.x1, x.y1, x.x2, x.y2), x.conf) for x in Detector(a.weights, conf=a.conf).detect(page)]
    texts = ocr_page(np.asarray(page))
    texts += ocr_inside_parts(np.asarray(page), parts, texts)
    net = extract_netlist(np.asarray(page), parts, texts)

    vf = vlm_check.find_vlm_file(a.input, a.page, [ROOT / "inputs" / "vlm"])
    be = None
    if vf is None and a.vlm != "none":
        try:
            be = (vlm_check.OllamaBackend(a.vlm_model or vlm_check.OLLAMA_MODEL) if a.vlm == "ollama"
                  else vlm_check.HFBackend(a.vlm_model or vlm_check.HF_MODEL))
        except Exception as e:
            print(f"vision model not available ({e}); using the chip OCR only")
    vlm_check.second_opinion(page, net, be, vf, list(Taxonomy.load().classes))

    out = a.out / f"{a.input.stem}_p{a.page}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "netlist.json").write_text(json.dumps(net), encoding="utf-8")
    info = make_dsn(net, out, a.input.stem, pdf=True)
    low_p = [p for p in net["parts"] if p["conf"] < VERIFY_BELOW]
    low_n = [n for n in net["nets"] if len(n["pins"]) >= 2 and n["conf"] < VERIFY_BELOW]
    lines = [f"{a.input.name} page {a.page}", f"parts {len(net['parts'])}, nets {len(net['nets'])}, drawing style {info.get('style')}, "
             f"OrCAD check {'passed' if info.get('verified') or info.get('style') == 'labels' else 'FAILED'} "
             f"({info.get('pins_checked')} pins, {info.get('pins_mismatched')} mismatched)",
             f"to check by a person: {len(low_p)} parts, {len(low_n)} nets (under {VERIFY_BELOW:.0%})", ""]
    lines += [f"part {p['id']} {p['cls']} {p['conf']:.0%}: {p.get('vlm_flag') or ''}" for p in low_p]
    lines += [f"net {n['id']} {n['name']} {n['conf']:.0%}: {(n['doubts'] or [{}])[0].get('what', '')}" for n in low_n]
    (out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:3]))
    print("files:", out)


if __name__ == "__main__":
    main()
