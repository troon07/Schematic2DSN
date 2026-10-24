"""Builds notebooks/train_supermodel_colab.ipynb (run: python notebooks/make_super_notebook.py)."""

import json
from pathlib import Path

MD_INTRO = """# Super detector: all datasets, long training, survives disconnects

**Every session: Runtime -> Run all.** That is the whole routine. Whatever is already done is skipped:
datasets come back from Drive in minutes, and training continues from the last finished epoch
(same optimizer and learning-rate schedule, so the result is the same as one uninterrupted run).

**Once, before the first run**, in Google Drive folder `T2/` (the same folder as before):
- `code.zip` (new one from this project), `Dataset300.zip`, `ci2n_datasets-main.zip` (already there)
- Runtime -> Change runtime type -> **GPU** (T4 works; L4 / A100 are 2-6x faster)

**Data used** (all public): PCBnet Dataset300, Image2Net, CGHD (hand-drawn, 4.9 GB), Digitize-HCD
(hand-drawn, 1,277 scans), Open Schematics (real KiCad project sheets, `OS_SHARDS` x ~70 pages) and
KiCad-rendered synthetic pages for every class incl. logic gates, relays, optocouplers.

**Drive space**: about 4-6 GB for the prepared data + ~0.3 GB for checkpoints.

**Time**: one epoch is roughly 15-25 min on a T4 with the defaults (printed after epoch 1).
300 epochs is many sessions; the run also stops by itself when the validation score stops improving
for 60 epochs. You can stop any time: `T2/super_runs/super/weights/best.pt` is always the best so far.
"""

SETUP = r"""# 1. Setup (every session, ~2 min)
from google.colab import drive
drive.mount('/content/drive')
DRIVE = '/content/drive/MyDrive/T2'
import os, subprocess, time
%cd /content
!mkdir -p proj/data/raw && unzip -q -o $DRIVE/code.zip -d proj && echo 'code unpacked'
%cd /content/proj
!pip -q install ultralytics pymupdf pyarrow > /dev/null && echo 'packages ok'
!apt-get -qq install -y aria2 > /dev/null && echo 'aria2 ok'
import torch
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else
      'NONE - Runtime > Change runtime type > T4 GPU. If none is offered, the free GPU time is used up: try later.')
"""

SETTINGS = r"""# 2. Settings. Change BEFORE the first training run; a started run keeps its own settings.
MODEL = 'yolo11m.pt'   # yolo11s (fast) / yolo11m (default) / yolo11l (A100 / L4)
EPOCHS = 300           # upper limit; stops earlier when no longer improving (patience 60)
IMGSZ = 1280           # schematics have small parts: keep 1280
OS_SHARDS = 150        # Open Schematics shards (~70 real sheets each); 0 = skip
SYN_PAGES = 8000       # synthetic KiCad pages (all 47 classes)
USE_CGHD, USE_DHCD = True, True
"""

DATA = r"""# 3. Datasets: built once, kept on Drive (T2/super_data), restored in later sessions
CACHE, Y = f'{DRIVE}/super_data', '/content/proj/data/yolo'
os.makedirs(CACHE, exist_ok=True); os.makedirs(Y, exist_ok=True)

def sh(cmd):
    print('$', cmd[:160], flush=True)
    if subprocess.run(cmd, shell=True).returncode:
        raise RuntimeError('failed: ' + cmd[:160])

def cached(name, build):
    tar, done, ready = f'{CACHE}/{name}.tar', f'{CACHE}/{name}.done', f'{Y}/{name}/.ready'
    if os.path.exists(ready):
        return print(f'{name}: ready')
    t = time.time()
    if os.path.exists(done):
        sh(f'tar -xf "{tar}" -C {Y}')
    else:
        build()
        sh(f'tar -cf /content/{name}.tar -C {Y} {name} && cp /content/{name}.tar "{tar}.tmp" && mv "{tar}.tmp" "{tar}" '
           f'&& touch "{done}" && rm /content/{name}.tar')
    open(ready, 'w').close()
    print(f'{name}: done in {(time.time() - t) / 60:.1f} min', flush=True)

def build_detector():
    sh(f'unzip -q -o {DRIVE}/Dataset300.zip -d data/raw -x "__MACOSX/*"')
    sh(f'unzip -q -o {DRIVE}/ci2n_datasets-main.zip -d data/raw -x "ci2n_datasets-main/device_orientation/*"')
    sh('python scripts/prepare_detector_data.py | tail -n 3')

def build_cghd():
    sh('aria2c -x 16 -s 16 -c --summary-interval=30 --console-log-level=warn -d /content -o cghd.zip '
       '"https://zenodo.org/records/15333233/files/cghd-zenodo-15.zip?download=1"')
    sh('unzip -q -o /content/cghd.zip -d /content/cghd_raw && rm -f /content/cghd.zip')
    sh(f'python scripts/prepare_cghd.py --src /content/cghd_raw --out {Y}/cghd | tail -n 3 && rm -rf /content/cghd_raw')

def build_dhcd():
    sh('aria2c -x 16 -s 16 -c --summary-interval=30 --console-log-level=warn -d /content -o dhcd.zip '
       '"https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/c4ec3d20-38ea-4f1d-96f9-cd504e998a6e"')
    sh(f'python scripts/prepare_digitize_hcd.py --zip /content/dhcd.zip --out {Y}/digitize_hcd && rm -f /content/dhcd.zip')

cached('detector', build_detector)
cached('synthetic', lambda: sh(f'python scripts/make_synthetic.py --pages {SYN_PAGES} --out {Y}/synthetic | tail -n 1'))
if USE_CGHD: cached('cghd', build_cghd)
if USE_DHCD: cached('digitize_hcd', build_dhcd)
for start in range(0, OS_SHARDS, 50):            # 50 shards per chunk: a disconnect loses at most one chunk
    n = min(50, OS_SHARDS - start)
    cached(f'open_schematics_{start:03d}', lambda start=start, n=n: sh(
        f'python scripts/prepare_open_schematics.py --start {start} --shards {n} '
        f'--out {Y}/open_schematics_{start:03d} --raw /content/os_raw --debug 0 | tail -n 2'))
sh(f'python scripts/make_combined.py --out {Y}/super.yaml')
sh(f'du -sh "{CACHE}"')
"""

TRAIN = r"""# 4. Train - or continue after a disconnect (just run it again)
!python scripts/train_super.py --data {Y}/super.yaml --sync {DRIVE}/super_runs --model {MODEL} --epochs {EPOCHS} --imgsz {IMGSZ}
"""

PROGRESS = r"""# 5. Progress so far (works while training runs in another session too)
import pandas as pd
p = f'{DRIVE}/super_runs/super/results.csv'
if os.path.exists(p):
    df = pd.read_csv(p); df.columns = [c.strip() for c in df.columns]
    print(f"{len(df)} epochs done; best mAP50 {df['metrics/mAP50(B)'].max():.3f}, best mAP50-95 {df['metrics/mAP50-95(B)'].max():.3f}")
    display(df[['epoch', 'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']].tail(10))
else:
    print('no epochs finished yet')
"""

EVAL = r"""# 6. Honest scores: every source's held-out split separately (needs cell 3 in this session)
!python scripts/eval_super.py --weights {DRIVE}/super_runs/super/weights/best.pt --imgsz {IMGSZ}
"""

MD_USE = """## Use the model on your PC

Download `T2/super_runs/super/weights/best.pt` and put it at `D:/T2/runs/detector/weights/best.pt`
(keep the old file under another name). The dashboard and scripts pick it up automatically.

**Reading the scores (cell 6):** mAP50 is the detection score per source. Hand-drawn photos (CGHD,
Digitize-HCD) are the hardest; printed / KiCad pages score highest. The weakest classes listed per source
are where more data helps most: reviewed pages from the dashboard (`prepare_retrain.py`) can be added later.
"""


def cell(kind, src):
    c = {"cell_type": kind, "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}
    if kind == "code":
        c.update({"execution_count": None, "outputs": []})
    return c


nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"}},
      "cells": [cell("markdown", MD_INTRO), cell("code", SETUP), cell("code", SETTINGS), cell("code", DATA),
                cell("code", TRAIN), cell("code", PROGRESS), cell("code", EVAL), cell("markdown", MD_USE)]}
out = Path(__file__).with_name("train_supermodel_colab.ipynb")
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out)
