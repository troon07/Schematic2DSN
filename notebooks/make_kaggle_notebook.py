"""Builds notebooks/train_supermodel_kaggle.ipynb (run: python notebooks/make_kaggle_notebook.py)."""

import json
from pathlib import Path

MD_INTRO = """# Super detector on Kaggle (2 x T4): long training in 11-hour pieces

**Every run: Save Version -> Save & Run All (Commit).** It runs on Kaggle's servers for up to ~11 h even
if you close the browser, stops itself cleanly after the last epoch that fits, and keeps everything in
its **Output**. The next run takes that Output as input and continues from the last epoch.

**Once, before the first run**
1. kaggle.com -> your profile -> Settings: verify your phone number (needed for GPU and internet).
2. Datasets -> New Dataset, name it `t2-files`, upload `code.zip`, `Dataset300.zip`, `ci2n_datasets-main.zip`
   (Kaggle may unpack them; that is fine). Private.
3. *Optional, to continue the Colab training instead of starting over:* from Google Drive
   `T2/super_runs/super/` download `weights/last.pt`, `weights/best.pt`, `results.csv`, `args.yaml`,
   and make a dataset `t2-colab-run` with the same layout: `super_runs/super/weights/last.pt`, ...
4. New Notebook -> File -> Import notebook -> this .ipynb. Right panel: **Add Input** -> `t2-files`
   (and `t2-colab-run` if you made it). **Settings -> Accelerator: GPU T4 x2**, **Internet: on**.

**Every later run:** Add Input -> Your Work -> this notebook's **latest output** (it holds the checkpoint and the
prepared datasets), then Save Version -> Save & Run All. Remove older outputs from the inputs.

Kaggle gives about 30 GPU hours per week; each run uses up to ~11.5 of them.
"""

SETUP = r"""# 1. Setup (~3 min)
import os, glob, time, subprocess, zipfile, shutil
SESSION_START = time.time()
IN = '/kaggle/input'
OUT = '/kaggle/working'
def find(name):
    hits = glob.glob(f'{IN}/**/{name}', recursive=True)
    return hits[0] if hits else None
def sh(cmd):
    print('$', cmd[:160], flush=True)
    if subprocess.run(cmd, shell=True).returncode:
        raise RuntimeError('failed: ' + cmd[:160])

os.makedirs('/content/proj/data/raw', exist_ok=True)
code = find('code.zip')
if code:
    sh(f'unzip -q -o "{code}" -d /content/proj')
else:                                     # Kaggle unpacked code.zip: copy the folders
    src = os.path.dirname(os.path.dirname(find('circuit2dsn/__init__.py')))
    for d in ('circuit2dsn', 'configs', 'scripts'):
        shutil.copytree(f'{src}/{d}', f'/content/proj/{d}', dirs_exist_ok=True)
    shutil.copy(f'{src}/requirements.txt', '/content/proj/') if os.path.exists(f'{src}/requirements.txt') else None
os.chdir('/content/proj')
sh('pip -q install ultralytics pymupdf pyarrow psutil > /dev/null')
sh('apt-get -qq install -y aria2 > /dev/null || true')
import torch
print('GPUs:', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] or
      'NONE - Settings -> Accelerator -> GPU T4 x2')
"""

SETTINGS = r"""# 2. Settings. Change BEFORE the first run; a started run keeps its own settings.
MODEL = 'yolo11m.pt'
EPOCHS = 300
IMGSZ = 1280
OS_SHARDS = 150
SYN_PAGES = 8000
USE_CGHD, USE_DHCD = True, True
HOURS = 11.3          # Kaggle stops a run at 12 h; the run stops itself before that and saves
"""

DATA = r"""# 3. Datasets: restored from an attached earlier output, else built (first run ~1 h), then saved to Output
CACHE, Y = f'{OUT}/super_data', '/content/proj/data/yolo'
os.makedirs(CACHE, exist_ok=True); os.makedirs(Y, exist_ok=True)

def cached(name, build):
    ready = f'{Y}/{name}/.ready'
    if os.path.exists(ready):
        return print(f'{name}: ready')
    t = time.time()
    mine = f'{CACHE}/{name}.tar'
    prev = [p for p in glob.glob(f'{IN}/**/super_data/{name}.tar', recursive=True)]
    if prev:
        sh(f'tar -xf "{prev[0]}" -C {Y}')
        if not os.path.exists(mine):
            sh(f'cp "{prev[0]}" "{mine}"')          # keep it in this run's Output for the next run
    else:
        build()
        sh(f'tar -cf "{mine}.tmp" -C {Y} {name} && mv "{mine}.tmp" "{mine}"')
    open(ready, 'w').close()
    print(f'{name}: done in {(time.time() - t) / 60:.1f} min', flush=True)

def unpack(name, dest):
    z = find(name + '.zip')
    if z:
        sh(f'unzip -q -o "{z}" -d {dest} -x "__MACOSX/*"')
    else:                                  # Kaggle unpacked it
        d = glob.glob(f'{IN}/**/{name}', recursive=True)
        sh(f'cp -r "{d[0]}" {dest}/')

def build_detector():
    unpack('Dataset300', 'data/raw')
    unpack('ci2n_datasets-main', 'data/raw')
    sh('python scripts/prepare_detector_data.py | tail -n 3')

def build_cghd():
    sh('aria2c -x 16 -s 16 -c --summary-interval=60 --console-log-level=warn -d /tmp -o cghd.zip '
       '"https://zenodo.org/records/15333233/files/cghd-zenodo-15.zip?download=1"')
    sh('unzip -q -o /tmp/cghd.zip -d /tmp/cghd_raw && rm -f /tmp/cghd.zip')
    sh(f'python scripts/prepare_cghd.py --src /tmp/cghd_raw --out {Y}/cghd | tail -n 3 && rm -rf /tmp/cghd_raw')

def build_dhcd():
    sh('aria2c -x 16 -s 16 -c --summary-interval=60 --console-log-level=warn -d /tmp -o dhcd.zip '
       '"https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/c4ec3d20-38ea-4f1d-96f9-cd504e998a6e"')
    sh(f'python scripts/prepare_digitize_hcd.py --zip /tmp/dhcd.zip --out {Y}/digitize_hcd && rm -f /tmp/dhcd.zip')

cached('detector', build_detector)
cached('synthetic', lambda: sh(f'python scripts/make_synthetic.py --pages {SYN_PAGES} --out {Y}/synthetic | tail -n 1'))
if USE_CGHD: cached('cghd', build_cghd)
if USE_DHCD: cached('digitize_hcd', build_dhcd)
for start in range(0, OS_SHARDS, 50):
    n = min(50, OS_SHARDS - start)
    cached(f'open_schematics_{start:03d}', lambda start=start, n=n: sh(
        f'python scripts/prepare_open_schematics.py --start {start} --shards {n} '
        f'--out {Y}/open_schematics_{start:03d} --raw /tmp/os_raw --debug 0 | tail -n 2'))
sh(f'python scripts/make_combined.py --out {Y}/super.yaml')
sh(f'du -sh {OUT}')
"""

TRAIN = r"""# 4. Train / continue. Stops by itself before the time limit, right after a saved epoch.
left = HOURS - (time.time() - SESSION_START) / 3600
print(f'{left:.1f} h of training in this run')
sh(f'python scripts/train_super.py --data {Y}/super.yaml --sync {OUT}/super_runs --local /content/runs '
   f'--resume-from "{IN}/*/super_runs" "{IN}/*/*/super_runs" --model {MODEL} --epochs {EPOCHS} --imgsz {IMGSZ} '
   f'--device auto --max-hours {left:.2f}')
"""

PROGRESS = r"""# 5. Progress so far
import pandas as pd
p = f'{OUT}/super_runs/super/results.csv'
if os.path.exists(p):
    df = pd.read_csv(p); df.columns = [c.strip() for c in df.columns]
    print(f"{len(df)} epochs done; best mAP50 {df['metrics/mAP50(B)'].max():.3f}, best mAP50-95 {df['metrics/mAP50-95(B)'].max():.3f}")
    print(df[['epoch', 'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']].tail(10).to_string(index=False))
"""

EVAL = r"""# 6. Honest scores per source (held-out test pages)
if os.path.exists(f'{OUT}/super_runs/super/weights/best.pt'):
    sh(f'python scripts/eval_super.py --weights {OUT}/super_runs/super/weights/best.pt --imgsz {IMGSZ}')
"""

MD_USE = """## Get the model

In this run's **Output** (right panel, or the notebook's Output tab after the run): `super_runs/super/weights/best.pt`.
Download it to `D:/T2/runs/detector/weights/best.pt` (keep the old file under another name).
"""


def cell(kind, src):
    c = {"cell_type": kind, "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}
    if kind == "code":
        c.update({"execution_count": None, "outputs": []})
    return c


nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}},
      "cells": [cell("markdown", MD_INTRO), cell("code", SETUP), cell("code", SETTINGS), cell("code", DATA),
                cell("code", TRAIN), cell("code", PROGRESS), cell("code", EVAL), cell("markdown", MD_USE)]}
out = Path(__file__).with_name("train_supermodel_kaggle.ipynb")
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out)
