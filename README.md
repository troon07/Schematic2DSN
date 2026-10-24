<p align="center">
  <h1 align="center">🔧 Schematic2DSN</h1>
  <p align="center"><b>Turn circuit schematics into OrCAD Capture <code>.dsn</b> files</b></p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12-blue?logo=python" alt="Python 3.12"/>
  <img src="https://img.shields.io/badge/YOLO11-detection-green?logo=ultralytics" alt="YOLO11"/>
  <img src="https://img.shields.io/badge/OrCAD-Capture-orange?logo=cadence" alt="OrCAD Capture"/>
  <img src="https://img.shields.io/badge/EDIF-2.0.0-lightgrey" alt="EDIF 2.0.0"/>

  <img src="https://img.shields.io/badge/platform-Windows-lightgrey" alt="Windows"/>
</p>

---

## 📋 What it does

**Schematic2DSN** (`circuit2dsn`) converts circuit diagrams — PDF pages, scanned images, photos, or hand drawings — into production-ready **OrCAD Capture `.dsn` schematic files** via **EDIF 2.0.0**. It detects components, reads labels, traces wires, builds nets, and lets a human review anything the model is unsure about before export.

```
PDF / PNG / JPG
    │
    ▼
YOLO11 detection  →  OCR + text cleanup  →  Wire tracing  →  Pin-to-net
    │                                                              │
    └────────────── Review UI (approve / fix / split / merge) ─────┘
                                │
                                ▼
                        EDIF → OrCAD Capture .dsn
                        (verified pin-by-pin by OrCAD itself)
```

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎯 **Component detection** | YOLO11 tiled detection, 47 component classes, per-part confidence bands (high / medium / low) |
| 🔤 **Text & OCR** | PP-OCRv4 (RapidOCR), role-aware cleanup, designator/value/net-label/pin-number matching |
| 🔌 **Wire tracing** | Ink mask → skeleton → junction rules (dot / crossing / hop / T / L) with per-junction confidence |
| 🕸️ **Pin-to-net construction** | Wire-end pin finding, touching-part contacts, ground/power/net-label merging, confidence + doubts per net |
| 🧩 **Module grouping** | Bridge rectifier, crystal + caps, LED + R, divider, RC filter, decoupling, pull-ups — approve as one block |
| 👁️ **Review UI** | Browser-based (port 8765), approve/fix/delete/add parts and connections, undo (50 steps), 70% verify gate |
| 📄 **OrCAD Capture output** | Real symbols + routed wires + power symbols + ports + footprints, EDIF 2.0.0, verified pin-by-pin by OrCAD |
| 📊 **Industry NetList.xml** | TD9 TestSequencer / SDDEMO schema export (`_NetList.xml`) for industry test-sequencer import |
| 📑 **Multi-page PDF** | Nets join across pages by GND / power / port names and net labels |
| 🔄 **Retraining** | Export reviewed pages as YOLO labels + corrections for the next training round |
| 📦 **Outputs** | `.dsn`, `.edf`, `.cir` (SPICE), `_bom.csv`, `_netlist.json`, `_connections.png`, `_NetList.xml`, `_all.zip` |

---

## 📊 Model accuracy & results

### Detector (trained model — `runs/detector/weights/best.pt`)

| Metric | Value |
|--------|-------|
| Architecture | YOLO11m (super detector) |
| Classes | 47 |
| mAP50 | **94.1%** |
| mAP50-95 | **84.5%** |
| Precision | **95.5%** |
| Recall | **91.2%** |
| Epochs | 12 |

*Previous baseline (YOLO11s, 28 classes): mAP50 78.7%, fitness 0.560.*

### Wire tracing & netlist (PCBnet / Dataset300, annotated parts)

| Level | Exact pins | Pair precision | Pair recall |
|-------|------------|----------------|-------------|
| Wire networks (pins from boxes) | 82.7% | 86.0% | 82.4% |
| Electrical nets (+ GND, power, net labels) | 49.4% (val 57.4%) | 77.8% | 75.3% |

### Text (PCBnet test split, annotated text boxes)

| Step | Result |
|------|--------|
| OCR exact, raw RapidOCR | 26.5% |
| OCR exact, tuned crops + cleanup | 66.7% (designators 88.8%, values 75.0%, net labels 60.3%) |
| Designator → part | 82.8% (val 85.3%) |
| Value → part | 85.8% (val 82.9%) |
| Power / port symbol name | 91.2% (val 83.7%) |
| Nets: no text / OCR text / annotated text | 47.4% / 49.4% / 51.2% |

### Pin labels (PCBnet test, annotated text)

| Metric | Accuracy |
|--------|----------|
| Pin names | 90.3% |
| Pin numbers | 72.6% |
| Pin names (OCR text) | 66.8% |
| Pin numbers (OCR text) | 50.5% |

### OrCAD Capture verification (gold standard)

| Test | Result |
|------|--------|
| Designs verified by OrCAD | **60 / 60** (30 test + 30 val) |
| Pins checked | **7,300** |
| Pin mismatches | **0** |
| Import errors | **0** |
| Example2 (photo, full pipeline) | 43 parts, 26 nets, 100/100 pins verified, 0 errors |
| Example1 (review end-to-end) | 304 pins verified by OrCAD, 0 mismatches |
| Multi-page (10 synthetic 3-page docs) | 0 errors, 0 mismatches |

---

## 🚀 Quick start (Windows)

```powershell
# 1. Create environment
python -m venv .venv
.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\pip install -r requirements.txt

# 2. Prepare detector data & smoke-test training
.venv\Scripts\python scripts\prepare_detector_data.py
.venv\Scripts\python scripts\train_detector.py --smoke

# 3. Detect components
.venv\Scripts\python scripts\detect.py my_circuit.pdf

# 4. Launch review dashboard
.venv\Scripts\python scripts\review.py
# Open http://127.0.0.1:8765 in your browser
```

*No NVIDIA GPU on this laptop? `--smoke` only checks that training works. For real training use `notebooks/train_detector_colab.ipynb` on a free Colab T4 GPU and copy `best.pt` to `runs/detector/weights/`.*

---

## 🖥️ Running the dashboard

```powershell
.venv\Scripts\python scripts\review.py
```

Open **http://127.0.0.1:8765**, then:
1. Pick **Review** or **Fully automatic** mode
2. Drop a **PDF / PNG / JPG** schematic
3. Approve components and connections (lowest confidence first)
4. Click **Export** → get `.dsn`, `.edf`, `.cir`, `_bom.csv`, `_netlist.json`, `_connections.png`, `_NetList.xml`, `_all.zip`

---

## 📁 Repository layout

```
circuit2dsn/          Core library
  detect.py           Tiled YOLO11 detection + confidence bands
  wires.py            Ink mask → skeleton → junction rules
  netlist.py          Pin finder, pins → wire groups, contacts
  pipeline.py         Page + parts + text → nets (confidence + doubts)
  textroles.py / textmatch.py / textfix.py / textstage.py  OCR text roles, matching, cleanup
  ocr.py              PP-OCRv4 recogniser (RapidOCR)
  modules.py          Circuit-block finder (rectifier, divider, RC, ...)
  multipage.py        Join per-page netlists
  orientation.py      Diode / BJT / MOSFET pin-role classifiers
  schematic.py        Netlist → drawn-schematic EDIF (symbols, router)
  edif.py             Netlist → compact EDIF (labels / joined styles)
  edif_check.py       EDIF parser + Capture-style connectivity check
  review_state.py     Review session: actions, undo, re-trace, export, training data
  unknowns.py         “Unrecognised drawing” marks
  orcad.py / dsn_export.py  EDIF → OrCAD .dsn via TCL, import + verification
  netlist_xml.py      Industry NetList.xml (TD9 TestSequencer / SDDEMO) exporter
  review_ui/index.html Browser review page
configs/
  classes.yaml        47-class taxonomy + dataset label mapping
  footprints.yaml     Default PCB footprint per class
scripts/              CLI entry points (detect, extract_netlist, export_edif, review, export_netlist_xml, ...)
notebooks/            Colab / Kaggle training notebooks
data/                 Datasets (PCBnet, Image2Net, CGHD, Digitize-HCD, Open Schematics)
```

---

## 🧠 Pipeline at a glance

1. **Page rasterisation** (PDF → 200 dpi RGB, or single image)
2. **Component detection** — YOLO11 whole-page + overlapping 1280 px tiles, class-aware NMS
3. **Text detection + OCR** — PP-OCRv4, role assignment (designator / value / net_label / pin_number / pin_name / part_name / other), role-aware cleanup
4. **Text → part matching** — nearest-text with prefix/unit consistency, alignment bonus, greedy assignment
5. **Wire tracing** — ink mask, erase bodies/text, skeletonise, junction rules, union-find → wire groups
6. **Pin finding** — wire ends on box edges + touching-part contacts
7. **Net building** — wires → ground → power/port names → net labels; every net carries `conf` + `doubts`
8. **Module grouping** — topology rules for batch approval
9. **Review UI** — approve/fix/split/merge; export with training data
10. **EDIF → OrCAD Capture** — real symbols + routed wires; OrCAD import + pin-by-pin verification; fallback to label-style EDIF so wrong connections are never shipped silently

---

## 📤 Outputs

From review export (`outputs/review/<name>/export/`):

| File | What it is |
|------|------------|
| `<name>.dsn` | OrCAD Capture schematic (verified) |
| `<name>.edf` | EDIF 2.0.0 interchange file |
| `<name>.cir` | SPICE netlist |
| `<name>_bom.csv` | Parts list / BOM |
| `<name>_netlist.json` | Machine-readable netlist |
| `<name>_connections.png` | Connection overlay image |
| `<name>_NetList.xml` | Industry TD9 TestSequencer / SDDEMO netlist |
| `<name>_all.zip` | Everything above in one zip |

---

## ⚠️ Known gaps

- Tiny text (~8 px in low-res renders) is the main OCR failure
- Parts drawn touching (e.g. GND symbol straight on an IC pin) need contact review
- Hand-drawn/scanned input benefits from a learned wire mask (in progress)
- Stock OrCAD libraries lack modern MCUs — add company `.olb` libraries via `--extra`
- LLM correction pass for low-confidence readings not yet built

---

## 🤝 Contributing

Pull requests welcome. Please open an issue first for major changes.

---

<details>
<summary>📄 Full project report</summary>

See [`PROJECT_REPORT.md`](PROJECT_REPORT.md) for the complete design document, stage-by-stage details, limitations, and future work.
</details>

---

<p align="center"><b>Schematic2DSN</b> — schematic to OrCAD Capture, verified pin-by-pin.</p>
