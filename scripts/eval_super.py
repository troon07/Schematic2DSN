"""Score a detector on every source's held-out split, separately.

    python scripts/eval_super.py --weights runs/super/weights/best.pt

One number over everything hides weak spots (hand-drawn photos vs printed
datasheets vs real KiCad sheets), so each source is scored on its own test
split (val where a source has no test split), plus per-class mAP50 and
precision / recall. Writes runs/<name>_eval.json next to the weights.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from circuit2dsn.taxonomy import Taxonomy  # noqa: E402
from scripts.make_combined import SOURCES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--base", type=Path, default=ROOT / "data" / "yolo")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--target", type=float, default=0.95)
    args = ap.parse_args()
    import torch
    from ultralytics import YOLO

    if args.device != "cpu" and not torch.cuda.is_available():
        print("No GPU in this session: scoring on the CPU instead (slower, same numbers).")
        args.device, args.batch = "cpu", 2
    names = Taxonomy.load().classes
    model = YOLO(str(args.weights))
    report = {}
    tmp = Path(tempfile.mkdtemp())
    for src in SOURCES:
        dirs = [d for d in sorted(args.base.glob(f"{src}*"))      # chunked sources: open_schematics_000 ...
                if d.is_dir() and (d.name == src or d.name[len(src):].startswith("_"))]
        split = next((s for s in ("test", "val")
                      if any((d / "images" / s).exists() and any((d / "images" / s).iterdir()) for d in dirs)), None)
        if split is None:
            continue
        paths = [(d / "images" / split).resolve().as_posix() for d in dirs if (d / "images" / split).exists()]
        y = tmp / f"{src}.yaml"
        y.write_text("train:\n" + "".join(f"  - {p}\n" for p in paths) + "val:\n" + "".join(f"  - {p}\n" for p in paths)
                     + "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)))
        r = model.val(data=str(y), imgsz=args.imgsz, batch=args.batch, device=args.device, plots=False,
                      verbose=False, project=str(tmp), name=src)
        per = {names[int(c)]: round(float(r.box.ap50[k]), 3) for k, c in enumerate(r.box.ap_class_index)}
        report[src] = {"split": split, "mAP50": round(float(r.box.map50), 4), "mAP50-95": round(float(r.box.map), 4),
                       "precision": round(float(r.box.mp), 4), "recall": round(float(r.box.mr), 4), "per_class": per}
        flag = "meets" if r.box.map50 >= args.target else "below"
        print(f"\n== {src} ({split}): mAP50 {r.box.map50:.3f}  P {r.box.mp:.3f}  R {r.box.mr:.3f}  "
              f"-> {flag} the {args.target:.0%} target")
        weak = sorted(per.items(), key=lambda kv: kv[1])[:8]
        print("   weakest classes:", ", ".join(f"{k} {v:.2f}" for k, v in weak))
    out = args.weights.parent.parent / f"{args.weights.parent.parent.name}_eval.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"\nreport: {out}")


if __name__ == "__main__":
    main()
