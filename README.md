# circuit2dsn

Turn circuit diagrams (PDF schematics, scans, hand drawings) into OrCAD Capture
`.dsn` files. Output goes through **EDIF**, which OrCAD Capture imports
(File → Import Design → EDIF) and saves as `.dsn`.

Two modes (planned):
- **Auto**: detect everything, build the most likely netlist, write EDIF, and list low-confidence items in a report.
- **Review**: diagram on the left, questions on the right. Approve each component and connection, or approve whole modules (e.g. "D1–D4 = bridge rectifier") for big circuits. Corrections are saved as new training data.

## Pipeline and status

| Stage | What | Status |
|---|---|---|
| 1 | Component detector (YOLO11, 28 classes, tiled, per-part confidence) | **ready to train** |
| 2 | Text: detector with roles (YOLO), OCR (PP-OCRv4 via RapidOCR), cleanup, text-to-part matching | **OCR + matching working**; text detector **ready to train** |
| 3 | Wires: ink mask, skeleton, junction dot / crossing / hop rules | **working** (classical, no training) |
| 4 | Pins from box edges + wire groups → nets (ground, power names, net labels) | **working**, needs stage 2 text |
| 5 | Module grouping (rectifier, crystal, LED+R, divider, RC, decoupling, pull-ups) | **working** (rules) |
| 6 | EDIF writer → OrCAD Capture `.dsn` | **working**, self-checked; needs a first real import in Capture |
| 7 | Review UI (component / connection / module approval) | **working**, tested in a browser |
| 8 | Hand-drawn support (CGHD dataset + sketch augmentation) | todo |

## Layout

```
configs/classes.yaml           unified class list + mapping from each dataset's labels
circuit2dsn/taxonomy.py        loads the class map
circuit2dsn/detect.py          tiled detection, confidence bands, overlay drawing
circuit2dsn/wires.py           ink mask, text erase, skeleton graph, junction rules
circuit2dsn/netlist.py         pin finder (wire ends on box edges), pins -> wire groups
circuit2dsn/pipeline.py        page + parts (+ text) -> nets with conf and doubts
circuit2dsn/pcbnet.py          PCBnet ground truth loader, ground-truth nets
circuit2dsn/textroles.py       text roles (detector classes) + rules for untagged PCBnet text
circuit2dsn/ocr.py             recogniser on crops (no border, 3x upscale, wire lines removed)
circuit2dsn/textfix.py         role-aware OCR cleanup (10kn -> 10kΩ, CT -> C7, Chinese on digits)
circuit2dsn/textmatch.py       designator / value / symbol name -> part
circuit2dsn/textstage.py       boxes + roles -> recognised, cleaned text
scripts/prepare_detector_data.py   PCBnet + Image2Net -> YOLO dataset (tiling, splits)
scripts/train_detector.py      training (CPU smoke test or full GPU run)
scripts/detect.py              run on a PDF / image -> JSON + coloured overlay
scripts/extract_netlist.py     page -> netlist JSON + net overlay (--pcbnet <id> demo)
scripts/eval_wires.py          score wire tracing vs PCBnet (oracle pins / box pins)
scripts/eval_netlist.py        score full nets vs PCBnet (--text none|gt|ocr)
scripts/prepare_text_data.py   PCBnet texts -> YOLO text dataset (7 roles, same splits)
scripts/eval_ocr.py            OCR accuracy on annotated text boxes (--fix for cleanup)
scripts/eval_textmatch.py      text-to-part matching accuracy
circuit2dsn/edif.py            netlist JSON -> EDIF 2 0 0 schematic (grid-snapped, labels / joined styles)
circuit2dsn/edif_check.py      EDIF parser + Capture-style connectivity check
scripts/export_edif.py         netlist JSON -> .edf with self-check
scripts/preview_edif.py        draw an .edf to PNG (what Capture should show)
scripts/test_edif_roundtrip.py design -> netlist -> EDIF -> parse -> same nets? (30 test designs)
circuit2dsn/modules.py         circuit-block finder for module-wise approval
circuit2dsn/review_state.py    review session: actions, undo, re-trace, export, training data
circuit2dsn/review_ui/index.html  the review page (no build step, no extra installs)
scripts/review.py              starts the review page on a local server
circuit2dsn/multipage.py       join per-page netlists into one design (GND, power, ports, labels)
circuit2dsn/orientation.py     diode / BJT / MOSFET pin roles from two small classifiers
scripts/combine_pages.py       combine reviewed pages into one multi-page netlist + EDIF
scripts/prepare_retrain.py     your reviewed pages -> extra training data (--zip for Colab)
scripts/prepare_orientation_data.py, scripts/train_orientation.py   orientation classifiers (CPU)
scripts/eval_pinlabels.py      pin number / name matching accuracy
notebooks/train_detector_colab.ipynb   full training on a free Colab/Kaggle GPU
data/raw/                      extracted datasets (not edited)
data/yolo/detector/            generated training set
runs/                          training outputs, weights
```

## Data

- **PCBnet / Dataset300**: 300 real schematics (KiCad, Altium, JLC, datasheets) with components, pins, texts, wires, SPICE netlists and EDIF. Split 80/10/10 per design.
- **Image2Net / ci2n**: 2,269 analog diagrams with component boxes. Split 85/10/5. Its NPN/PNP and NMOS/PMOS labels are kept as `subtype` in `annotations.jsonl` for a later crop classifier.
- Pages larger than 1600 px are cut into 1280 px tiles with 25% overlap for train/val. Test pages stay whole.

## Quick start (Windows, this machine)

```bash
python -m venv .venv
.venv/Scripts/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python scripts/prepare_detector_data.py
.venv/Scripts/python scripts/train_detector.py --smoke
```

This laptop has no NVIDIA GPU, so `--smoke` only checks that training works.
For real training, use `notebooks/train_detector_colab.ipynb` on a free T4 GPU
and copy `best.pt` back to `runs/detector/weights/`. Then:

```bash
.venv/Scripts/python scripts/detect.py my_circuit.pdf
```

Outputs go to `outputs/detect/`: a JSON file with every component, its box and
confidence, and an overlay image (green ≥ 80%, orange 50–80%, red < 50%).

## Wires and netlist (stages 3-4)

`scripts/extract_netlist.py` runs: ink mask → erase text and part bodies →
1 px skeleton → junction rules → pins where wires end on a part's box edge →
nets. Every net has `conf` (its weakest link) and `doubts` (which junction,
pin or name merge is uncertain), for the review mode.

Junction rules: a dot (blob thicker than `1.42 × stroke + 1.5 px`, i.e. thicker
than any crossing of two lines) connects; four wires meeting without a dot is
a crossing, including curved hop arcs; T and L shapes connect.

```bash
.venv/Scripts/python scripts/extract_netlist.py --pcbnet 0002   # demo with annotated parts
.venv/Scripts/python scripts/eval_wires.py --mode boxes
.venv/Scripts/python scripts/eval_netlist.py
```

Results on the 30 PCBnet test designs, using annotated parts and text in place
of the detector and OCR. "Exact" = share of pins whose set of connected pins is
exactly right.

| Level | Exact pins | Pair precision | Pair recall |
|---|---|---|---|
| Wire networks, pins found from boxes | 82.7% | 86.0% | 82.4% |
| Electrical nets (+ ground, power names, net labels) | 49.4% (val 57.4%) | 77.8% | 75.3% |

On validation, nets flagged doubtful are 31.5% correct, against 77% for the
rest, so the doubt flag points at real errors.

Known gaps:
- **Parts drawn touching** (e.g. a GND symbol placed straight on an IC pin): no wire is left after erasing bodies.
- **Pin-number boxes** (TI datasheet style): the boxes chain an IC's pins together.
- **Text stage needed**: power names and net labels come from annotations now; at inference they need OCR plus text-to-part matching.
- **PCBnet labels are noisy**: some wire networks are split in fragments (merged where a pin is shared), and the SPICE files are generated from the same annotations.
- **Hand-drawn / scanned input** needs a learned wire mask; PCBnet edge `coordinates` can serve as training masks.

## Text (stage 2)

1. **Text detector** (YOLO, to train in the notebook's Part 2) finds text and
   its role: designator, value, net_label, pin_number, pin_name, part_name,
   other. PCBnet tags 60% of its texts; the rest get roles from rules
   (`textroles.py`), e.g. text linked to a power symbol is a net name.
2. **OCR** (`ocr.py`): PP-OCRv4 recogniser bundled with RapidOCR, on CPU.
   Settings found on PCBnet val: no white border around the crop (a 4 px
   border cut exact reads from 71% to 42%), 3x nearest-neighbour upscale,
   wire lines whitened. Decoding is not limited to ASCII because JLC
   schematics contain real Chinese text.
3. **Cleanup** (`textfix.py`) by role: Ω misread as n/Q/n2, look-alike
   digits in designators and pin numbers, stray brackets from wires,
   known net names (GND, VCC...) for low-confidence readings, and one
   canonical value form (0.1 μF = 0.1uF, 4K7 = 4.7kΩ).
4. **Matching** (`textmatch.py`): nearest text with designator-prefix and
   value-unit consistency, alignment bonus, one text per part.

PCBnet test split, annotated text boxes:

| Step | Result |
|---|---|
| OCR exact match, raw RapidOCR | 26.5% |
| OCR exact, tuned crops + cleanup | 66.7% (designators 88.8%, values 75.0%, net labels 60.3%) |
| Designator → part | 82.8% (val 85.3%) |
| Value → part | 85.8% (val 82.9%) |
| Power/port symbol name | 91.2% (val 83.7%) |
| Nets, exact pins: no text / OCR text / annotated text | 47.4% / 49.4% / 51.2% |

Text adds about 4 points at net level; wire-level errors and unattached
pins limit the net score more. With OCR, nets flagged doubtful are 39%
correct against 65% for the rest.

Known gaps: tiny text (~8 px tall in low-resolution renders) is still the
main OCR failure; PCBnet's own text labels contain OCR-style errors
(`I014` vs `IO14`, `10F` for `10μF`). An LLM correction pass for
low-confidence readings (as in the PCBnet paper) is not built yet.

## OrCAD Capture output (stage 6)

The dashboard export (and `circuit2dsn.dsn_export.make_dsn`) produces a ready
`.dsn` with OrCAD's own importer, prints it to PDF and checks it with OrCAD's
own netlist. Needs OrCAD on the PC (`D:	ools`, or set `ORCAD_TOOLS`).

**What the .dsn contains** (`circuit2dsn/schematic.py`)
- **Real symbols**: standard symbols per class (bundled KiCad library: US
  zig-zag resistor, capacitor, diode, LED, NPN/PNP, MOSFET, op-amp, gates,
  relay, optocoupler ...), rotated/mirrored so the pins point where the wires
  met the part, matching detected pin roles (diode K=1 A=2, BJT B=1 C=2 E=3).
  ICs and unknown parts are boxes with one named pin per detected pin.
- **Real wires**: every wire island of the drawing is re-routed on the 100 mil
  grid (A* following the original wire path). Different nets never touch;
  they only cross at right angles. Capture adds junction dots at T-joins.
  Pins the router cannot reach (about 0.5%) get a short wire with a net alias.
- **Power symbols and ports**: ground / supply symbols become Capture power
  objects (GND, VCC bar), terminals become ports (input / output / bidirectional
  by name), written the way Capture itself exports them.
- **PCB Footprint** on every part from `configs/footprints.yaml` (edit to your
  library names); sources/meters are marked PSpiceOnly.
- Unique designators: unread ones continue the numbering (R21, Q18 ...),
  property C2D_AUTO_REF=TRUE marks them.
- Import settings: `circuit2dsn/orcad_edi2cap.cfg` (100 mil grid fixed; the
  default "calculated" grid would rescale the design).

**Verification**: after import, OrCAD exports the .dsn's connectivity
(XMATIC_CAP2EDIF) and every pin is compared with the netlist. If anything
differs, the export falls back to the compact label style (parts as boxes,
every pin a named stub; exact by construction), so a wrong connection is
never shipped silently.

`scripts/test_dsn_schematic.py` (PCBnet designs, annotated parts):
**60/60 designs (30 test + 30 val) verified by OrCAD, 0 of 7,300 pins
different, 0 import errors.** Example2 (photo, full detector + OCR pipeline):
43 parts, 26 nets, 100/100 pins verified, 0 errors, 0 warnings.

Manual import of an `.edf` still works: File > Import Design > EDIF tab.

## Review mode (stage 7)

```bash
.venv/Scripts/python scripts/review.py --pcbnet 0006        # demo, no trained models needed
.venv/Scripts/python scripts/review.py my_circuit.pdf --page 1  # after training both detectors
.venv/Scripts/python scripts/review.py --resume outputs/review/my_circuit_p1
```

Opens http://127.0.0.1:8765 in your browser (local only). The diagram is on
the left (scroll to zoom, drag to pan) with the current item highlighted;
the question is on the right.

1. **Components**: "U19 · detected as other. Is that right?" Yes / Fix
   (type, designator, value, net name) / Not a part / Skip. Approve a whole
   type at once ("27 × ground: All correct"), approve everything above 85%,
   or draw a box around a missed part (A). Lowest confidence comes first.
2. **Connections**: wires are re-traced with the checked parts. Circuit
   blocks come first ("Voltage divider R6 / R5" ✓ confirms all its
   connections). Then each net, lowest confidence first, drawn as lines
   between its pins with the reason for any doubt. Click a pin that does not
   belong to split it off, join another net, rename, or approve.
3. **Export**: corrected netlist JSON + EDIF (self-checked) + training data
   (the page with corrected YOLO labels and a list of text fixes) in
   `outputs/review/<name>/`.

Keys: Y yes · E fix · N not a part · S skip · U undo · F zoom to item · A add part · Esc cancel.
Every click is saved to `session.json`; undo keeps 50 steps.

Block rules (`modules.py`, topology only): bridge rectifier (4 diodes in a
ring), crystal + load caps, LED + series resistor, voltage divider, RC
filter, decoupling caps per supply, pull-ups per supply. On the 30 test
designs they find 88 blocks.

## Pin labels, touching parts, multi-page, retraining, orientation

**Pin numbers and names** (`textmatch.match_pins`): a pin name sits on the
pin's line just inside the body; a pin number sits beside the stub, offset
to one side. That offset is found per side of each part (above / centred /
below), so a number between two close pins goes to the right one. PCBnet
test, annotated text: names 90.3%, numbers 72.6% correct (OCR text: 66.8% /
50.5%). In the EDIF, ports carry the pin name and the pin number; missing or
duplicate numbers fall back to the next free number.

**Parts drawn touching** (`netlist.find_contacts`): where two part boxes
touch or overlap slightly and there is ink in the shared strip, the parts
connect directly. Pairs of single-pin symbols (stacked ports, net ties,
power/ground) and resistor pairs are excluded: on PCBnet they touch without
connecting more often than not. Val: exact pins 58.3% -> 59.9%, recall 67.8%
-> 72.3%, precision unchanged; test: about the same, unattached pins 204 -> 179.
Contacts carry confidence 0.6, so review asks about them.

**Pin-number boxes (TI datasheet style)**: tried growing a part over drawn
boxes around pin numbers; it hurt PCBnet results (val exact pins 59.9% ->
55.4%) because a number squeezed between two stubs and the body outline
looks "boxed". Removed. Review mode can split such pins.

**Multi-page PDFs**: `extract_netlist.py` also writes `<name>_all_netlist.json`
and `<name>_all.edf` (one EDIF page per PDF page). Nets join across pages by
GND, power/port names and net labels (`--page-scope power` joins only
power/ground/ports). `combine_pages.py` does the same for reviewed pages.
Tested on 10 synthetic 3-page documents: 0 errors, 0 mismatches.

**Retraining**: `prepare_retrain.py` adds every exported review page (x5) to
the detector data; `--zip` packs them for the notebook's Part 3 fine-tune.
Reviews of PCBnet val/test designs are skipped so demos never leak into
training.

**Orientation**: `prepare_orientation_data.py` + `train_orientation.py`
train two tiny classifiers on Image2Net crops on the CPU (flips off: a
flipped diode is the opposite diode). Pin roles: diode/LED A/K, BJT B/C/E,
MOSFET G. Used automatically when `runs/orient/weights/best.pt` exists; in
the EDIF they name the pins and give standard numbers (K=1 A=2; B=1 C=2
E=3; G=1) when the drawing's own numbers were not read.

## Dashboard: Review or Fully automatic

```bash
.venv/Scripts/python scripts/review.py
```

Open http://127.0.0.1:8765, pick a mode, drop a PDF / PNG / JPG.

- **Review**: check components, then connections, then export (as above).
- **Fully automatic**: finds components, reads labels, traces wires and
  writes every file at once (~30 s for a 1700 px page on this PC). The
  results page shows the parts and nets on the drawing, a parts table, a
  clickable net list, and downloads:
  `.edf` (OrCAD Capture EDIF), `.cir` (SPICE), `_bom.csv` (parts list),
  `_connections.png`, `_netlist.json`, and `_all.zip`.
  "Review it" turns the result into a review.
  Automatic results are never used as training data (they are unchecked).

**Text without the trained text model**: RapidOCR's own text finder reads the
page (angle classifier off: it flipped "Output"), roles come from the words
(`textroles.role_from_content`), and a second zoomed pass reads designators
drawn inside symbols ("Q8" between a transistor's leads). Designators and
values stacked in one label block are matched to the same part.

**Detection clean-up**: a box lying mostly inside a more confident box is
dropped (half-boxes of one part from tiles).

Example: `inputs/example2.jpg` (LM317-style regulator, 43 parts): 43 parts
found, all 26 connected nets checked by hand and correct (1 stray pin), 32 of
43 designators read; misses: Q7, Q9, Q14, R7, R8, R9, D1, D2 (fix in review).

## Super detector: all datasets, long resumable training

`notebooks/train_supermodel_colab.ipynb` (regenerate with
`python notebooks/make_super_notebook.py`). Every session: Runtime > Run all.

- Data: PCBnet + Image2Net, CGHD, **Digitize-HCD** (1,277 hand-drawn scans,
  `scripts/prepare_digitize_hcd.py`, reads only the images out of the 2.1 GB
  zip), **Open Schematics** (real KiCad sheets; exact boxes computed from each
  `.kicad_sch`, `scripts/prepare_open_schematics.py`), KiCad synthetic pages
  (`scripts/make_synthetic.py`, parallel, resumable).
- Each dataset is built once and stored on Drive (`T2/super_data/*.tar` +
  `.done`); later sessions restore it in minutes.
- `scripts/train_super.py` trains on local disk and copies the run to Drive
  after every epoch (temp file + rename, previous copy kept as last_prev.pt).
  Rerunning resumes from the last epoch; a damaged checkpoint falls back to the
  previous one; a finished run is reported, not restarted. Tested locally:
  interrupted run, lost VM, corrupted checkpoint, finished run.
- `scripts/eval_super.py` scores each source's held-out split separately
  (mAP50, precision, recall, weakest classes) against the 95% target.

## OrCAD parts, unrecognised drawing, 70% rule

**Every OrCAD library part in review.** `python scripts/build_orcad_catalog.py`
(once, ~30 min; `--extra D:/CompanyLibs` adds the company's own `.olb` files)
has OrCAD export all 322 installed libraries: **73,686 parts** with pins,
pin numbers, symbol drawing, reference prefix and footprint
(`data/orcad_catalog/`). In the dashboard, "OrCAD part…" opens a search box
(part number or type: LM358, 1N4148, 7805, relay ...). The chosen part is
written into the .dsn with OrCAD's own symbol, pin numbers and footprint
(`circuit2dsn/orcad_parts.py`), and checked by OrCAD's netlist like every
other part. Stock OrCAD libraries do not contain modern MCUs (e.g. STM32):
add the company libraries with `--extra`.

**Unrecognised drawing** (`circuit2dsn/unknowns.py`): loose wire ends that
touch no part, and curved / slanted strokes on the circuit's wires that no
detected part covers, are marked as red dashed "?" boxes. Example2 (all parts
found): 0 marks. Example1 (old 28-type model, misses gates): marks on the
missed NOR / XOR / AND / OR gates, LED, phototransistor, optocoupler,
crystal, variable capacitor and terminals. The reviewer adds the part there
(type or OrCAD part) or dismisses it.

**70% rule** (`VERIFY_BELOW` in `review_state.py`): parts and connections
under 70% confidence plus every unrecognised mark must be checked by a
person, also in fully automatic mode. Automatic mode exports at once but
marks the files **draft** and opens the "Check uncertain" step; after the
checks, "Make final export" re-traces the wires with the added parts (keeping
approvals of unchanged connections) and writes the final, OrCAD-verified
.dsn. Tested end to end on example1: 36 items to check -> final export,
304 pins verified by OrCAD, 0 mismatches.

## Wiring fixes for tight-box drawings (junction classifier, adaptive tracing)

Symptoms seen on Image2Net-style pages: a capacitor or MOSFET gate showing two pins on one net, voltage sources with both pins on one wire, a divider crossing that did not join, a port left on a scrap of its own outline.

- `circuit2dsn/junction_model.py` + `models/junction.pt`: small CNN deciding whether a 4-way crossing is a dot (connected) or two wires passing. Trained from the jumper labels (`scripts/prepare_junction_data.py`, `scripts/train_junction.py`). On validation 4-arm crossings: dots caught 68.5% -> 94.0%, accuracy 89.5% -> 98.2%.
- `circuit2dsn/tracing.py`: each part is traced with a safe erase margin first, then checked against what its symbol must look like (pin count, 2-terminal parts must not have both pins on one wire). Failing parts are re-traced with their body fully erased; the tighter erase is kept only where it came out better. Capacitor/battery plates at the box rim are merged into one pin. A port/ground/power symbol whose pin sits on a wire reaching no other part is re-attached to the connected wire touching its box.
- `scripts/eval_pins_proxy.py`: ground-truth-free consistency benchmark (pin count right, duplicate pins, lone one-pin symbols). Image2Net: count_ok 71.2 -> 80.9%, duplicates 24.7 -> 15.4%.
- PCBnet regression: pin_exact test 82.7 -> 85.7%, val 85.2 -> 87.6%; high-confidence nets correct test 67.0 -> 71.3%, val 75.5 -> 78.5%.
- Existing review sessions keep their old results: re-process the page (or press "Done · trace wires") and restart `scripts/review.py`.

## Second opinion from a local vision model, and real symbols in the .dsn

- `circuit2dsn/vlm_check.py` (+ `scripts/vlm_verify.py`, `COLAB_VLM_CHECK.txt`): Qwen2.5-VL-7B reads crops of uncertain parts and ICs (part number, reference), is judged against the detector (`judge()`: "unknown" = no opinion; an IC with an OrCAD part number counts as agreement), and matches the number to the OrCAD libraries (exact, `A` revision, or "did you mean"). Agreeing ICs get their OrCAD symbol automatically; disagreements go to the 70% verify list.
- Automatic pass in the dashboard: `python scripts/review.py --vlm ollama` (after `ollama pull qwen2.5vl:7b`; slow on a CPU, fast on a GPU) checks parts, unidentified drawing and doubtful junctions after the detector. Or put Colab results (`<sheet>_p<N>.vlm.json`) in `inputs/vlm/`.
- .dsn symbols: parts with a standard symbol keep it when a few extra pins were detected (extras dropped, lone/least-sure first); OrCAD parts never get wires on their VCC/GND pins (Capture joins those itself) and extra detections are dropped. Before this, the Capture connectivity check failed on IC-heavy sheets and the exporter silently fell back to boxes with name stubs. All 30 PCBnet designs and the 5 company sheets now verify with 0 pin mismatches.
