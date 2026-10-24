# Project Report — circuit2dsn

**Turning Circuit Schematics into OrCAD Capture `.dsn` Files**

| Field | Value |
|---|---|
| Project name | circuit2dsn |
| Working directory | `D:\T2` |
| Language / stack | Python 3.12, PyTorch, Ultralytics YOLO11, OpenCV, scikit-image, RapidOCR (PP-OCRv4), PyMuPDF, OrCAD Capture (TCL) |
| Primary output | OrCAD Capture `.dsn` schematic (via EDIF 2.0.0) + `.cir` SPICE netlist + BOM CSV |
| Report date | 2026-10-23 |

---

## 1. Abstract

**circuit2dsn** is an end-to-end system that converts a circuit schematic — a PDF page, a scanned image, or a hand drawing — into a fully connected, OrCAD Capture-compatible `.dsn` schematic file. It combines (a) a trained component detector (YOLO11), (b) a text detector, OCR and role-aware cleanup, (c) classical computer-vision wire tracing, (d) pin-to-net construction with confidence and doubt tracking, (e) module/block recognition for batch approval, (f) an interactive review UI, and (g) an OrCAD-driven EDIF→`.dsn` conversion with a pin-by-pin connectivity verification. Every decision carries a confidence; uncertain items are surfaced for human review rather than shipped silently. On 60 PCBnet designs the pipeline achieves **0 pin mismatches out of ~7,300 pins** when verified by OrCAD's own netlist export, and the fully automatic mode on a real photo (`example2.jpg`) produced 43 parts / 26 nets with 100/100 pins verified.

---

## 2. Problem Statement and Motivation

Digitising a schematic by hand is slow and error-prone: a single page may contain hundreds of components, hundreds of wire junctions, and dozens of net labels. Existing tools either require manual re-drawing or fail on real-world inputs (scans, mixed font sizes, curved hop arcs, touching parts). The goal of this project is to automate the conversion while keeping a human in the loop for the parts the machine is unsure about, and to guarantee that what is shipped to OrCAD Capture is electrically correct (verified by OrCAD itself, not by the converter's own assumptions).

Key challenges addressed:

- **Scale variation**: a 3000–5000 px page contains both large ICs and 20 px ground symbols.
- **Wire semantics**: junction dots vs. crossings vs. hops must be distinguished without training data.
- **Text roles**: the same glyph string can be a designator, a value, a pin number, a pin name, or a net label depending on position and linked part.
- **OCR on tiny text**: ~8 px glyphs in low-resolution renders.
- **Electrical correctness**: a single wrong wire merges two nets; the system must never silently ship that.
- **Reviewability**: every uncertain item must be individually approvable, editable, or dismissible, with undo.

---

## 3. Objectives

1. Detect components (28 → 60 classes) on full-resolution tiled pages with per-part confidence.
2. Trace wires (ink mask → skeleton → junction rules) and build pins/nets with confidence + doubt metadata.
3. Read and clean text (designators, values, net labels, pin numbers/names) and attach it to parts.
4. Recognise common circuit blocks (rectifier, divider, RC, decoupling, pull-up, crystal, LED+R) for batch approval.
5. Export a ready-to-use OrCAD Capture `.dsn` (real symbols, routed wires, power symbols, ports, footprints) verified pin-by-pin.
6. Provide a review UI and a fully automatic mode, with a 70% confidence gate ("draft" vs "final").
7. Support multi-page PDFs, touching parts, polarised-part orientation, and retraining from corrections.

---

## 4. System Architecture

### 4.1 High-level pipeline

```
Input (PDF / PNG / JPG)
   │
   ▼
[1] Page rasterisation (PyMuPDF, 200 dpi)  ──► list of RGB page images
   │
   ▼
[2] Component detection (YOLO11, tiled, class-aware NMS)  ──► parts with boxes + conf
   │
   ▼
[3] Text detection (YOLO text model or RapidOCR fallback) + roles
   │
   ▼
[4] OCR (PP-OCRv4 via RapidOCR) + role-aware cleanup (textfix)  ──► texts with roles
   │
   ▼
[5] Text → part matching (designator / value / symbol name / pin number / pin name)
   │
   ▼
[6] Wire tracing (ink mask → skeleton → junction rules)  ──► wire groups
   │
   ▼
[7] Pin finding (wire ends on box edges) + contact detection (touching parts)
   │
   ▼
[8] Net building (wire groups + ground + power/port names + net labels)  ──► nets with conf + doubts
   │
   ▼
[9] Module/block recognition (topology rules)
   │
   ▼
[10] Review UI (optional)  or  Fully automatic export
   │
   ▼
[11] EDIF writer (symbols, wires, power, ports, footprints, refdes)  +  OrCAD import + verification
   │
   ▼
Outputs: .dsn, .edf, .cir (SPICE), _bom.csv, _connections.png, _netlist.json, _all.zip
```

### 4.2 Repository layout

```
circuit2dsn/            core library
  taxonomy.py           unified class list + dataset label mapping
  detect.py             tiled detection, confidence bands, overlay
  wires.py              ink mask, text erase, skeleton graph, junction rules
  netlist.py            pin finder, pins -> wire groups, contacts
  pipeline.py           page + parts + text -> nets with conf + doubts
  pcbnet.py             PCBnet (Dataset300) ground-truth loader
  textroles.py          text roles + rules for untagged PCBnet text
  ocr.py                recogniser on crops
  textfix.py            role-aware OCR cleanup
  textmatch.py          designator/value/symbol-name -> part; pin labels
  textstage.py          boxes + roles -> recognised, cleaned text
  modules.py            circuit-block finder
  multipage.py          join per-page netlists
  orientation.py        diode/BJT/MOSFET pin roles from two small classifiers
  schematic.py          netlist -> drawn schematic EDIF (symbols, wires, router)
  edif.py               netlist -> compact EDIF (labels/joined styles)
  edif_check.py         EDIF parser + Capture-style connectivity check
  review_state.py       review session: actions, undo, re-trace, export, training data
  unknowns.py           "unrecognised drawing" marks
  orcad.py              EDIF -> OrCAD .dsn via TCL (import/export/PDF)
  orcad_parts.py        searchable OrCAD part catalogue + symbol templates
  dsn_export.py         one-call verified .dsn (+PDF+PNG)
  review_ui/index.html  review page (no build step)
  kicad_symbols/        bundled KiCad symbol library (rendered for templates + synthetic data)
  kicad_render.py       render KiCad symbols to PIL images
configs/
  classes.yaml          unified taxonomy + source_map (PCBnet/ci2n/cghd)
  footprints.yaml       default PCB footprint per class
scripts/                CLI entry points (detect, extract_netlist, export_edif, review, ...)
notebooks/              Colab/Kaggle training notebooks + generators
data/                   datasets (raw + prepared YOLO sets)
runs/                   training outputs, weights
outputs/                detection/netlist/export artefacts
```

### 4.3 Core data structures

- **`Part`** (`pipeline.py`): `id, cls, box, conf, name`.
- **`Text`**: `content, box, role, conf`.
- **`Pin`** (`netlist.py`): `id, comp, x, y, group, conf`.
- **`WireGraph`** (`wires.py`): `seg_labels, group_of_seg, junctions, stroke, skeleton, endpoints, ink`.
- **`Netlist` (JSON)**: `parts, pins, nets, junctions, texts, stroke, unidentified, wires, width, height, pages`.
- **`Net`**: `id, name, pins, conf, doubts`.
- **`Detection`** (`detect.py`): `id, cls, conf, x1..x2` with `band` (high/medium/low).

---

## 5. Detailed Stage Descriptions

### 5.1 Stage 1 — Component detection (`detect.py`, YOLO11)

A schematic page is often 3000–5000 px wide while a ground symbol or 0402 capacitor is ~20 px, so a single resized pass misses small parts. The detector therefore runs on (a) the whole page (large parts like ICs) and (b) overlapping 1280 px tiles at native resolution (small parts), then merges with **class-aware NMS** (`batched_nms` from torchvision). A final `_drop_contained` pass removes boxes that lie ≥ 70% inside a more confident box (tile-edge half-boxes). Confidence bands: **high ≥ 0.80, medium 0.50–0.80, low < 0.50**. Output is JSON + a coloured overlay (green/orange/red).

### 5.2 Stage 2 — Text: detection, OCR, cleanup, matching

**Roles** (`textroles.ROLES`): `designator, value, net_label, pin_number, pin_name, part_name, other`. PCBnet tags ~60% of texts; the rest get roles by rules (e.g. text linked to a power symbol is a net name). A content-only fallback (`role_from_content`) exists for text found without the trained text model.

**OCR** (`ocr.py`): PP-OCRv4 recogniser bundled with RapidOCR, on CPU. Key tuned settings (validated on PCBnet val):
- No white border around the crop (a 4 px border cut exact reads 71% → 42%).
- 3× nearest-neighbour upscale.
- Wire lines whitened (a wire touching a label reads as `[` or `]`).
- Vertical labels read in both 90° rotations; more confident wins.
- `ascii_only` off by default (JLC schematics contain real Chinese text).

**Cleanup** (`textfix.py`): role-aware. Ω misread as n/Q/n2; look-alike digits in designators/pin numbers; stray brackets from wires; known net names for low-confidence readings; one canonical value form (`0.1 μF` = `0.1uF`, `4K7` = `4.7kΩ`).

**Matching** (`textmatch.py`): nearest text in the part's neighbourhood (gap scaled by part size), category consistency (designator prefix / value unit must suit the part), alignment bonus, global greedy assignment (one text per part, one part per text). Pin labels (`match_pins`) use per-side offset detection (above/centred/below) so a number between two close pins goes to the right one.

### 5.3 Stage 3 — Wires (`wires.py`)

Classical line-art pipeline, no training:

1. **Ink mask**: pixels differing from the dominant background colour.
2. **Erase component bodies** (otherwise a resistor's zig-zag shorts its pins) and **text glyphs** (keeping long horizontal/vertical runs = wires, and junction dots).
3. **Skeletonise** to 1 px lines; find junction clusters and segments.
4. **Junction rules**:
   - Junction dot (blob thicker than `1.42 × stroke + 1.5 px`) → all arms connected.
   - Four wires meeting without a dot → crossing (not connected): only opposite arms join; curved hop arcs handled.
   - T and L shapes → all arms connected.
   - Corner (2 arms) → connected.
5. **Union-find** over segments → wire groups.

Every decision records a confidence; doubtful junctions (e.g. borderline crossings) are reported.

### 5.4 Stage 4 — Pins and nets (`netlist.py`, `pipeline.py`)

**Pin finding** (`find_pins`): wire-end pixels lying on a component box edge become that component's pins. Wire ends of tiny stubs (leftover tips of erased symbol lines) are filtered out unless they join two parts that touch. For single-pin symbols (ground/power/port/terminal/no_connect) the best pins (longest wires) are kept.

**Contacts** (`find_contacts`): where two part boxes touch or overlap slightly and there is ink in the shared strip, the parts connect directly (confidence 0.6). Stacked single-pin symbols and resistor pairs are excluded (measured to touch without connecting more often than not). Contacts carry confidence 0.6 so review asks about them.

**Net building** (`pipeline.extract_netlist`), in order:
1. Wires: pins joined by drawn wire (wire groups).
2. Ground: every ground symbol is one net (GND).
3. Power/port symbols: same name → same net.
4. Net labels: wires carrying the same label text are one net (only `net_label` role merges; values/designators must not merge unrelated nets).

Each net carries `conf` (weakest link) and `doubts` (which junction/pin/name merge is uncertain).

### 5.5 Stage 5 — Module grouping (`modules.py`)

Topology-only rules (pin order of 2-pin parts unknown):
- Bridge rectifier (4 diodes forming a ring over 4 nets)
- Crystal + load caps
- LED + series resistor
- Voltage divider
- RC filter
- Decoupling caps per supply
- Pull-ups per supply

A part is used by at most one module. On 30 test designs they find 88 blocks. Modules come first in review ("Voltage divider R6/R5 ✓ confirms all its connections").

### 5.6 Stage 6 — OrCAD Capture output (`schematic.py`, `edif.py`, `orcad.py`, `dsn_export.py`)

Two writers:

- **`schematic.py` (drawn style)**: real symbols (bundled KiCad library: US zig-zag resistor, capacitor, diode, LED, NPN/PNP, MOSFET, op-amp, gates, relay, optocoupler…), rotated/mirrored so pins point where wires met the part and matching detected pin roles (diode K=1 A=2, BJT B=1 C=2 E=3). ICs and unknown parts are boxes with one named pin per detected pin. Wires are re-routed on the 100 mil grid (A* following the original wire path); different nets never touch, only cross at right angles. Power symbols and ports become Capture power objects/ports. PCB footprint on every part from `configs/footprints.yaml`; sources/meters marked PSpiceOnly. Unique designators (unread ones continue numbering; `C2D_AUTO_REF=TRUE` marks them). Pins the router cannot reach (~0.5%) get a short wire with a net alias.

- **`edif.py` (compact label style)**: every connected pin gets a 1-grid wire stub carrying the net's name; Capture joins same-named labels. Connectivity exact by construction. Two sub-styles: `labels` (each pin its own net+stub) and `joined` (one EDIF net per net).

**Verification**: OrCAD imports the `.edf` (`XMATIC_EDIF2CAP`), exports it back (`XMATIC_CAP2EDIF`), and every pin is compared with the netlist (`compare_nets`). If anything differs, export falls back to the compact label style, so a wrong connection is never shipped silently. `dsn_export.make_dsn` orchestrates: try drawn style, then labels style, import, verify, print to PDF, make PNG preview.

### 5.7 Stage 7 — Review UI (`review_state.py`, `review_ui/index.html`, `scripts/review.py`)

Local server on `http://127.0.0.1:8765`. Diagram on the left (scroll to zoom, drag to pan) with the current item highlighted; question on the right.

1. **Components**: "U19 · detected as other. Is that right?" Yes / Fix (type, designator, value, net name) / Not a part / Skip. Approve a whole type at once ("27 × ground: All correct"), approve everything above 85%, or draw a box around a missed part (A). Lowest confidence first.
2. **Connections**: wires re-traced with checked parts. Circuit blocks first ("Voltage divider R6/R5" ✓ confirms all its connections). Then each net, lowest confidence first, drawn as lines between its pins with the reason for any doubt. Click a pin that doesn't belong to split it off, join another net, rename, or approve.
3. **Export**: corrected netlist JSON + EDIF (self-checked) + training data in `outputs/review/<name>/`.

Keys: Y yes · E fix · N not a part · S skip · U undo · F zoom to item · A add part · Esc cancel. Every click saved to `session.json`; undo keeps 50 steps.

### 5.8 Hand-drawn / scanned support (status: partial)

- **CGHD** dataset (1,277 hand-drawn scans) and **Digitize-HCD** are integrated into the super detector training.
- **Unknowns** (`unknowns.py`): loose wire ends touching no part, and curved/slanted strokes on wires not covered by a detected part, are marked as red dashed "?" boxes. Reviewer adds the part (type or OrCAD part) or dismisses. Example1 (old 28-type model): marks on missed gates/LED/phototransistor/optocoupler/crystal/variable capacitor/terminals.
- A learned wire mask for hand-drawn input is still TODO (PCBnet edge `coordinates` can serve as training masks).

### 5.9 Multi-page PDFs (`multipage.py`, `extract_netlist.py`)

`extract_netlist.py` also writes `<name>_all_netlist.json` and `<name>_all.edf` (one EDIF page per PDF page). Nets join across pages by GND, power/port names and net labels (`--page-scope power` joins only power/ground/ports). `combine_pages.py` does the same for reviewed pages. Tested on 10 synthetic 3-page documents: 0 errors, 0 mismatches.

### 5.10 Orientation classifiers (`orientation.py`)

Two tiny classifiers (Image2Net crops, CPU, flips off because a flipped diode is the opposite diode):
- `orient`: diode_{u,d,l,r} = way the triangle points (cathode side); bjt_{..} = side of base; mos_{..} = side of gate.
- `mirror`: on a BJT crop rotated base-left, m0 = emitter on upper leg, m1 = lower.

Pin roles then follow from pin positions relative to part centre. Standard pin numbers (KiCad Device library order) used only when the drawing's own numbers were not read. Used automatically when `runs/orient/weights/best.pt` exists.

### 5.11 OrCAD part catalogue (`orcad_parts.py`, `build_orcad_catalog.py`)

`build_orcad_catalog.py` exports all 322 installed OrCAD libraries: **73,686 parts** with pins, pin numbers, symbol drawing, reference prefix and footprint (`data/orcad_catalog/`). In the dashboard, "OrCAD part…" opens a search box. The chosen part is written into the `.dsn` with OrCAD's own symbol, pin numbers and footprint, and checked by OrCAD's netlist like every other part. Stock OrCAD libraries lack modern MCUs (e.g. STM32): add company libraries with `--extra`.

### 5.12 The 70% rule (`review_state.VERIFY_BELOW`)

Parts and connections under 70% confidence plus every unrecognised mark must be checked by a person, also in fully automatic mode. Automatic mode exports at once but marks files **draft** and opens the "Check uncertain" step; after checks, "Make final export" re-traces wires with the added parts (keeping approvals of unchanged connections) and writes the final, OrCAD-verified `.dsn`. Tested end-to-end on example1: 36 items to check → final export, 304 pins verified by OrCAD, 0 mismatches.

---

## 6. Data and Datasets

| Dataset | Size | Content | Split |
|---|---|---|---|
| PCBnet / Dataset300 | 300 real schematics (KiCad, Altium, JLC, datasheets) | components, pins, texts, wires, SPICE netlists, EDIF | 80/10/10 per design |
| Image2Net / ci2n | 2,269 analog diagrams | component boxes; NPN/PNP and NMOS/PMOS kept as `subtype` | 85/10/5 |
| CGHD | hand-drawn scans | component boxes | (super detector) |
| Digitize-HCD | 1,277 hand-drawn scans | images only (read out of 2.1 GB zip) | (super detector) |
| Open Schematics | real KiCad sheets | exact boxes computed from `.kicad_sch` | (super detector) |
| KiCad synthetic | rendered pages | parallel, resumable generation (`make_synthetic.py`) | (super detector) |

Pages larger than 1600 px are cut into 1280 px tiles with 25% overlap for train/val; test pages stay whole. Unified taxonomy: **60 classes** (`configs/classes.yaml`), with `source_map` mapping each dataset's raw labels (PCBnet single letters, ci2n names, cghd dotted names) to `[class, subtype]`. Finer distinctions (NPN vs PNP, NMOS vs PMOS, bulk/cross variants) are kept as `subtype` for a later crop classifier; the detector stays coarse so datasets agree.

---

## 7. Training

### 7.1 Detector (YOLO11)

- `scripts/prepare_detector_data.py`: PCBnet + Image2Net → YOLO dataset (tiling, splits).
- `scripts/train_detector.py`: CPU smoke test (`--smoke`) or full GPU run.
- `notebooks/train_detector_colab.ipynb`: full training on free T4 GPU; copy `best.pt` to `runs/detector/weights/`.
- Super detector: `notebooks/train_supermodel_colab.ipynb` (regenerate with `make_super_notebook.py`). Each dataset built once and stored on Drive (`T2/super_data/*.tar` + `.done`); later sessions restore in minutes. `scripts/train_super.py` trains on local disk and copies the run to Drive after every epoch (temp file + rename; previous copy kept as `last_prev.pt`). Rerunning resumes from last epoch; damaged checkpoint falls back to previous; finished run reported, not restarted. Tested: interrupted run, lost VM, corrupted checkpoint, finished run.
- `scripts/eval_super.py` scores each source's held-out split separately (mAP50, precision, recall, weakest classes) against the 95% target.

### 7.2 Text detector

Trained in the notebook's Part 2; finds text and its role (7 roles). PCBnet tags 60% of texts; rest get roles from rules.

### 7.3 Orientation classifiers

`prepare_orientation_data.py` + `train_orientation.py`: two tiny classifiers on Image2Net crops, CPU (flips off). ~minutes.

### 7.4 Retraining from review

`prepare_retrain.py` adds every exported review page (×5) to detector data; `--zip` packs for the notebook's Part 3 fine-tune. Reviews of PCBnet val/test designs are skipped so demos never leak into training.

---

## 8. Evaluation and Results

### 8.1 Wire tracing (30 PCBnet test designs, annotated parts)

| Level | Exact pins | Pair precision | Pair recall |
|---|---|---|---|
| Wire networks, pins found from boxes | 82.7% | 86.0% | 82.4% |
| Electrical nets (+ ground, power names, net labels) | 49.4% (val 57.4%) | 77.8% | 75.3% |

Doubt flag points at real errors: nets flagged doubtful are 31.5% correct vs 77% for the rest.

### 8.2 Text (PCBnet test split, annotated text boxes)

| Step | Result |
|---|---|
| OCR exact match, raw RapidOCR | 26.5% |
| OCR exact, tuned crops + cleanup | 66.7% (designators 88.8%, values 75.0%, net labels 60.3%) |
| Designator → part | 82.8% (val 85.3%) |
| Value → part | 85.8% (val 82.9%) |
| Power/port symbol name | 91.2% (val 83.7%) |
| Nets exact pins: no text / OCR text / annotated text | 47.4% / 49.4% / 51.2% |

Text adds ~4 points at net level; wire-level errors and unattached pins limit the net score more.

### 8.3 Pin labels (PCBnet test, annotated text)

Names 90.3%, numbers 72.6% correct (OCR text: 66.8% / 50.5%).

### 8.4 Touching parts (contacts)

Val exact pins 58.3% → 59.9%, recall 67.8% → 72.3%, precision unchanged; test unattached pins 204 → 179.

### 8.5 OrCAD verification (`test_dsn_schematic.py`)

**60/60 designs (30 test + 30 val) verified by OrCAD, 0 of 7,300 pins different, 0 import errors.** Example2 (photo, full detector + OCR): 43 parts, 26 nets, 100/100 pins verified, 0 errors, 0 warnings.

### 8.6 Multi-page

10 synthetic 3-page documents: 0 errors, 0 mismatches.

### 8.7 End-to-end review (example1)

36 items to check → final export, 304 pins verified by OrCAD, 0 mismatches.

### 8.8 Example2 automatic

43 parts found, all 26 connected nets correct (1 stray pin), 32/43 designators read; misses: Q7, Q9, Q14, R7, R8, R9, D1, D2 (fix in review). 0 "unrecognised drawing" marks.

---

## 9. OrCAD Capture Integration

- OrCAD ships a TCL shell (`tools/bin/tclsh.exe`) and Capture's DB library (`orDb_Dll_Tcl64.dll`). The library provides `XMATIC_EDIF2CAP` (import), `XMATIC_CAP2EDIF` (export for verification), `XMATIC_DSN2PDF` (print), `XMATIC_DSN2XML` (net count).
- Install location: env var `ORCAD_TOOLS`, else common paths searched (`D:\tools`, `C:\Cadence\SPB_*\tools`, …).
- Import uses `orcad_edi2cap.cfg`: grid fixed to 100 mil (default "calculated" grid would rescale the design) and generated net names hidden.
- The `.dsn` format is Cadence's binary DB; only OrCAD writes it, so OrCAD must be installed on the same PC.

---

## 10. Dashboard (Review or Fully Automatic)

`scripts/review.py` (no args) opens `http://127.0.0.1:8765`, pick a mode, drop a PDF/PNG/JPG.

- **Review**: check components, then connections, then export (as above).
- **Fully automatic**: finds components, reads labels, traces wires, writes every file at once (~30 s for a 1700 px page on this PC). Results page shows parts and nets on the drawing, a parts table, a clickable net list, downloads: `.edf`, `.cir`, `_bom.csv`, `_connections.png`, `_netlist.json`, `_all.zip`. "Review it" turns the result into a review. Automatic results are never used as training data (unchecked).

Text without the trained text model: RapidOCR's own text finder reads the page (angle classifier off — it flipped "Output"), roles from words, plus a second zoomed pass for designators drawn inside symbols. Designators and values stacked in one label block are matched to the same part.

Detection clean-up: a box lying mostly inside a more confident box is dropped (half-boxes from tiles).

---

## 11. Limitations and Known Gaps

- Parts drawn touching (e.g. a GND symbol placed straight on an IC pin): no wire left after erasing bodies (mitigated by contact detection, conf 0.6).
- Pin-number boxes (TI datasheet style): attempted growing a part over drawn boxes around pin numbers; hurt PCBnet results (val exact pins 59.9% → 55.4%); removed. Review mode can split such pins.
- Text stage needed for power names and net labels at inference (OCR + text-to-part matching).
- PCBnet labels are noisy: some wire networks split into fragments (merged where a pin is shared); SPICE files generated from same annotations.
- Tiny text (~8 px tall in low-res renders) is the main OCR failure; PCBnet's own text labels contain OCR-style errors (`I014` vs `IO14`, `10F` for `10μF`).
- LLM correction pass for low-confidence readings (as in the PCBnet paper) not built yet.
- Hand-drawn/scanned input needs a learned wire mask (TODO).
- Stock OrCAD libraries lack modern MCUs; add company libraries.
- No NVIDIA GPU on this laptop: smoke tests only locally; real training on Colab/Kaggle.

---

## 12. Future Work

1. Learned wire mask for hand-drawn/scanned input (train on PCBnet edge coordinates).
2. LLM correction pass for low-confidence OCR readings.
3. Crop classifier for `subtype` (NPN/PNP, NMOS/PMOS, bulk/cross variants).
4. Larger, more diverse super-detector training to reach 95% mAP across all sources.
5. Pin-number box handling in review mode (split/merge).
6. Native `.dsn` reader/writer (currently relies on OrCAD).
7. Better handling of parts drawn touching (geometry-aware contact detection).

---

## 13. Quick Start (Windows, this machine)

```bash
python -m venv .venv
.venv/Scripts/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python scripts/prepare_detector_data.py
.venv/Scripts/python scripts/train_detector.py --smoke
```

No NVIDIA GPU → `--smoke` only checks training works. For real training use `notebooks/train_detector_colab.ipynb` on a free T4 GPU and copy `best.pt` back to `runs/detector/weights/`. Then:

```bash
.venv/Scripts/python scripts/detect.py my_circuit.pdf
```

Outputs go to `outputs/detect/`: JSON with every component (box + confidence) and an overlay (green ≥ 80%, orange 50–80%, red < 50%).

Wires and netlist:

```bash
.venv/Scripts/python scripts/extract_netlist.py --pcbnet 0002   # demo with annotated parts
.venv/Scripts/python scripts/eval_wires.py --mode boxes
.venv/Scripts/python scripts/eval_netlist.py
```

Review mode:

```bash
.venv/Scripts/python scripts/review.py --pcbnet 0006        # demo, no trained models needed
.venv/Scripts/python scripts/review.py my_circuit.pdf --page 1
.venv/Scripts/python scripts/review.py --resume outputs/review/my_circuit_p1
```

Dashboard:

```bash
.venv/Scripts/python scripts/review.py
```

---

## 14. Conclusion

circuit2dsn demonstrates that a hybrid approach — trained detection for parts/text plus classical CV for wires, confidence-tracked net construction, interactive review, and OrCAD-verified export — can reliably convert real schematics into production-ready OrCAD Capture designs. The verification step (OrCAD imports, re-exports, and every pin is compared) makes the system's correctness claim checkable rather than asserted: 60/60 designs, 0 of 7,300 pins different. The 70% confidence gate and the "unrecognised drawing" marks keep the human in the loop exactly where the machine is weak, and corrections flow back into training data, closing the loop.

---

## 15. References

- README.md (project) — pipeline, status, layout, quick start, metrics.
- PCBnet / Dataset300 — 300 real schematics with components, pins, texts, wires, SPICE netlists, EDIF.
- Image2Net / ci2n — 2,269 analog diagrams.
- CGHD, Digitize-HCD, Open Schematics — hand-drawn and real KiCad data for the super detector.
- Ultralytics YOLO11, RapidOCR (PP-OCRv4), PyMuPDF, scikit-image, OpenCV, OrCAD Capture TCL/DB API.
