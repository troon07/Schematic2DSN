"""One-click training on a local NVIDIA GPU (started by START_TRAINING.bat).

- checks the GPU, writes the dataset list for wherever this folder is now
- trains, continuing from RESULTS/ (or, the very first time, from
  start_checkpoint/ = the model trained so far)
- if training stops with an error, waits a minute and continues from the
  last saved epoch (up to 50 times)
- keeps Windows from sleeping while it runs (only while it runs; no setting
  is changed)
- everything is logged to RESULTS/training_log.txt

Backups (all in RESULTS/super/):
  weights/last.pt        newest epoch            weights/last_prev.pt  the one before
  weights/best.pt        best model so far       snapshots/epoch_NNN.pt every 5 epochs (6 kept)
"""

from __future__ import annotations

import ctypes
import datetime
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "RESULTS"
LOG = RESULTS / "training_log.txt"


def log(msg: str) -> None:
    line = f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    RESULTS.mkdir(exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def keep_awake(on: bool) -> None:
    if os.name == "nt":
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))


def main() -> int:
    import torch

    if not torch.cuda.is_available():
        log("NO NVIDIA GPU FOUND by PyTorch. Check that 'nvidia-smi' works in Command Prompt "
            "(NVIDIA driver installed). Training needs an NVIDIA GPU.")
        return 2
    gpu = torch.cuda.get_device_properties(0)
    log(f"GPU: {gpu.name}, {gpu.total_memory / 2**30:.1f} GB")
    vram = gpu.total_memory / 2**30
    imgsz = 1280 if vram >= 5.5 else 1024         # 4 GB cards: smaller images so it fits
    data = ROOT / "data" / "yolo" / "super.yaml"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_combined.py"), "--base", str(ROOT / "data" / "yolo"),
                    "--out", str(data)], check=True)
    cmd = [sys.executable, str(ROOT / "scripts" / "train_super.py"), "--data", str(data),
           "--sync", str(RESULTS), "--local", str(ROOT / "work" / "runs"),
           "--resume-from", str(ROOT / "start_checkpoint"),
           "--model", "yolo11m.pt", "--epochs", "300", "--imgsz", str(imgsz),
           "--device", "0", "--batch", "-1", "--workers", "4"]
    keep_awake(True)
    try:
        for attempt in range(1, 51):
            log(f"training start (attempt {attempt}); progress is saved after every epoch")
            with open(LOG, "a", encoding="utf-8") as lf:
                p = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, encoding="utf-8", errors="replace", bufsize=1)
                for line in p.stdout:
                    print(line, end="", flush=True)
                    if "[sync]" in line or "Error" in line or "error" in line.lower()[:40] or "Resuming" in line \
                            or "finished" in line or "Starting" in line:
                        lf.write(line)
                        lf.flush()
                code = p.wait()
            if code == 0:
                log("training finished (or nothing left to do). Model: RESULTS/super/weights/best.pt")
                return 0
            log(f"training stopped with an error (code {code}). Continuing from the last saved epoch in 60 s ...")
            time.sleep(60)
        log("stopped after 50 restarts: something keeps failing, send RESULTS/training_log.txt")
        return 1
    finally:
        keep_awake(False)


if __name__ == "__main__":
    sys.exit(main())
