"""Second-opinion check of a detect.py result with a vision-language model.

    python scripts/vlm_verify.py outputs/company/7_p1.json --backend hf        # Colab / any CUDA GPU
    python scripts/vlm_verify.py outputs/company/7_p1.json --backend ollama    # ollama pull qwen2.5vl:7b

Writes <name>.vlm.json next to the input (same format + a "vlm" entry per checked part).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from PIL import Image  # noqa: E402

from circuit2dsn import vlm_check  # noqa: E402
from circuit2dsn.taxonomy import Taxonomy  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("detections", type=Path, nargs="+")
    ap.add_argument("--backend", choices=["hf", "ollama"], default="hf")
    ap.add_argument("--model")
    ap.add_argument("--image", type=Path, help="the sheet image (use on Colab; the json holds a Windows path)")
    ap.add_argument("--images", type=Path, help="folder holding the sheets; each json is matched by name (7_p1.json -> 7.jpg)")
    ap.add_argument("--below", type=float, default=0.80, help="check every part less sure than this (ICs always)")
    a = ap.parse_args()
    classes = list(Taxonomy.load().classes)           # fail fast, before the big model loads
    be = (vlm_check.HFBackend(a.model or vlm_check.HF_MODEL) if a.backend == "hf"
          else vlm_check.OllamaBackend(a.model or vlm_check.OLLAMA_MODEL))
    for p in a.detections:
        if p.with_suffix(".vlm.json").exists():
            print(f"{p.name}: already done, skipped")
            continue
        d = json.loads(p.read_text())
        src = a.image or d["source"]
        if a.images:
            stem = p.name.split("_p")[0]
            src = next((f for f in a.images.glob(stem + ".*") if f.suffix.lower() in (".jpg", ".jpeg", ".png")), src)
        img = Image.open(src)
        vlm_check.verify(img, d["components"], be, classes, a.below,
                         progress=lambda i, n: print(f"\r{p.name}: {i}/{n}", end="", flush=True))
        out = p.with_suffix(".vlm.json")
        out.write_text(json.dumps(d, indent=1))
        ch = [c for c in d["components"] if "vlm" in c]
        dis = [c for c in ch if c["vlm"]["agrees"] is False]
        print(f"\n{out}: checked {len(ch)}, agree {sum(c["vlm"]["agrees"] is True for c in ch)}, could not tell {sum(c["vlm"]["agrees"] is None for c in ch)}, disagree {len(dis)}, "
              f"OrCAD-matched {sum('orcad' in c['vlm'] for c in ch)}, "
              f"unidentified {sum(c['vlm'].get('unidentified', False) for c in ch)}")


if __name__ == "__main__":
    main()
