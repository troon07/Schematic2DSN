"""Long detector training that survives disconnects (Google Colab or Kaggle).

Colab:   python scripts/train_super.py --data /content/proj/data/yolo/super.yaml \
             --sync /content/drive/MyDrive/T2/super_runs
Kaggle:  python scripts/train_super.py --data ... --sync /kaggle/working/super_runs \
             --resume-from "/kaggle/input/*/super_runs" --max-hours 11 --device auto

Run the same command again after every disconnect: it continues where it stopped.

How it is made safe:
- Training writes to the fast local disk (--local, default /content/runs).
- A watcher copies the run folder (last.pt, best.pt, results.csv, args.yaml)
  to --sync every time a new epoch is saved: temp file + rename, previous
  copy kept as last_prev.pt, so a break during a copy never leaves only a
  broken checkpoint. (A watcher, not a training callback, because with two
  GPUs Ultralytics trains in separate processes.)
- On start the newest checkpoint is taken from --sync, else from any
  --resume-from folder (e.g. the previous Kaggle run's output, or a copy of
  the Colab run). Paths stored inside it are updated to this machine, and
  Ultralytics resumes it: same epochs, optimizer, learning-rate schedule.
- --max-hours: stop right after the epoch that would run past the limit
  (Kaggle stops runs at 12 h; stopping first guarantees the save).
- --device auto: every GPU there is (Kaggle's 2 x T4 trains ~1.7x faster).
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _copy_atomic(src: Path, dst: Path, keep_prev: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copy2(src, tmp)
    if keep_prev and dst.exists():
        os.replace(dst, dst.with_name(dst.stem + "_prev" + dst.suffix))
    os.replace(tmp, dst)


def sync_run(local: Path, remote: Path) -> bool:
    """Copy what changed; True if a new last.pt went out."""
    sent = False
    for name in ("weights/last.pt", "weights/best.pt"):
        src = local / name
        if src.exists():
            dst = remote / name
            if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime + 1:
                _copy_atomic(src, dst, keep_prev=name.endswith("last.pt"))
                sent |= name.endswith("last.pt")
    for name in ("results.csv", "args.yaml", "results.png"):
        if (local / name).exists():
            _copy_atomic(local / name, remote / name)
    return sent


def _load(p: Path):
    try:
        import torch

        return torch.load(p, map_location="cpu", weights_only=False)
    except Exception as e:                       # noqa: BLE001 - a broken file: try the previous copy
        print(f"  {p} cannot be read ({type(e).__name__}); trying the next copy")
        return None


def restore(candidates: list[Path], local: Path, data: Path, project: Path):
    """Newest readable checkpoint among the candidate run folders -> local run
    folder, with its stored paths pointed at this machine. (path, ckpt) or (None, None)."""
    # the run's own folder (first candidate) always wins; the others (an
    # attached earlier run, a start checkpoint) only when it has nothing yet
    found = []
    for k, run in enumerate(candidates):
        for j, cand in enumerate(("weights/last.pt", "weights/last_prev.pt")):
            p = run / cand
            if p.exists():
                found.append(((0 if k == 0 else 1), j, -p.stat().st_mtime, run, p))
    for *_, run, p in sorted(found, key=lambda t: t[:3]):
        ck = _load(p)
        if ck is None:
            continue
        import torch

        ta = ck.get("train_args") or {}
        ta.update(data=str(data), project=str(project), name=local.name, save_dir=str(local), exist_ok=True)
        ck["train_args"] = ta
        (local / "weights").mkdir(parents=True, exist_ok=True)
        torch.save(ck, local / "weights" / "last.pt")
        for name in ("weights/best.pt", "results.csv", "args.yaml"):
            if (run / name).exists() and not (local / name).exists():
                shutil.copy2(run / name, local / name)
        print(f"Checkpoint restored from {p}")
        return local / "weights" / "last.pt", ck
    return None, None


def _epochs_done(local: Path) -> int:
    f = local / "results.csv"
    try:
        return max(0, len(f.read_text().strip().splitlines()) - 1)
    except OSError:
        return 0


def _kill_training():
    """Stop training processes (also the per-GPU workers) after a save."""
    try:
        import psutil

        me = psutil.Process()
        for c in me.children(recursive=True):
            c.kill()
    except ImportError:
        pass
    os._exit(0)


def snapshot(remote: Path, epoch: int, every: int, keep: int) -> None:
    """Every `every` epochs keep a full copy of last.pt (can restart training
    or be used as a model); only the newest `keep` are kept."""
    if every <= 0:
        return
    snaps = remote / "snapshots"
    snaps.mkdir(parents=True, exist_ok=True)
    have = [int(f.stem.split("_")[1]) for f in snaps.glob("epoch_*.pt")]
    if epoch < (max(have) if have else 0) + every:
        return
    _copy_atomic(remote / "weights" / "last.pt", snaps / f"epoch_{epoch:03d}.pt")
    old = sorted(snaps.glob("epoch_*.pt"))
    for f in old[:-keep]:
        f.unlink(missing_ok=True)


def watcher(local: Path, remote: Path, max_hours: float | None, t_start: float, stop: threading.Event,
            snap_every: int = 5, snap_keep: int = 6):
    last_sync, durations, first = t_start, [], True
    while not stop.wait(10):
        lp = local / "weights" / "last.pt"
        if not lp.exists():
            continue
        s1 = (lp.stat().st_mtime, lp.stat().st_size)
        time.sleep(3)
        if not lp.exists() or (lp.stat().st_mtime, lp.stat().st_size) != s1:     # still being written
            continue
        try:
            if not sync_run(local, remote):
                continue
        except OSError as e:
            print(f"[sync] failed ({e}); will retry", flush=True)
            continue
        now = time.time()
        if first:                       # the first copy is the restored checkpoint, not a trained epoch
            first = False
        else:
            durations.append(now - last_sync)
        last_sync = now
        ep = _epochs_done(local)
        print(f"[sync] epoch {ep} saved to {remote}", flush=True)
        try:
            snapshot(remote, ep, snap_every, snap_keep)
        except OSError as e:
            print(f"[sync] snapshot failed ({e})", flush=True)
        if max_hours and durations:
            per_epoch = sorted(durations)[len(durations) // 2]
            if now - t_start + 1.3 * per_epoch > max_hours * 3600:
                print(f"[stop] {max_hours} h limit: stopping after this saved epoch. "
                      f"Run again (with this output attached) to continue.", flush=True)
                _kill_training()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--model", default="yolo11m.pt", help="start weights for a NEW run (ignored when resuming)")
    ap.add_argument("--name", default="super")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=-1, help="-1 = as large as fits (one GPU); 4 per GPU with several")
    ap.add_argument("--device", default="auto", help="auto = every GPU; '0'; '0,1'; 'cpu'")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--local", type=Path, default=Path("/content/runs"))
    ap.add_argument("--sync", type=Path, required=True, help="folder that keeps the run (Drive / Kaggle output)")
    ap.add_argument("--resume-from", nargs="*", default=[], help="more run folders to resume from (globs allowed)")
    ap.add_argument("--max-hours", type=float, default=None, help="stop cleanly before this many hours")
    ap.add_argument("--snapshot-every", type=int, default=5, help="keep a full checkpoint every N epochs")
    ap.add_argument("--snapshot-keep", type=int, default=6)
    args = ap.parse_args()
    t_start = time.time()
    import torch

    n_gpu = torch.cuda.device_count()
    if args.device != "cpu" and n_gpu == 0:
        print("\nNO GPU IN THIS SESSION - nothing was lost, training was NOT started.\n"
              "The machine has no GPU (on free Colab: the GPU time for today is used up).\n"
              "  Colab:  Runtime > Change runtime type > T4 GPU > Save, then Run all.\n"
              "  Kaggle: Settings (right panel) > Accelerator > GPU T4 x2.\n"
              "The next GPU session continues from the last saved epoch.")
        return
    device = ",".join(str(i) for i in range(n_gpu)) if args.device == "auto" else args.device
    multi = "," in device
    batch = args.batch if args.batch >= 1 or not multi else 4 * len(device.split(","))   # 4 per T4 at 1280 px
    print(f"GPUs: {n_gpu} -> device {device}, batch {batch}")

    args.local, args.sync = args.local.resolve(), args.sync.resolve()      # Ultralytics nests relative paths
    local, remote = args.local / args.name, args.sync / args.name
    local.mkdir(parents=True, exist_ok=True)
    cands = [remote] + [Path(p) / args.name for pat in args.resume_from for p in glob.glob(pat)]
    last, ck = restore(cands, local, args.data.resolve(), args.local)

    stop = threading.Event()
    threading.Thread(target=watcher, args=(local, remote, args.max_hours, t_start, stop, args.snapshot_every,
                                          args.snapshot_keep), daemon=True).start()
    from ultralytics import YOLO

    if last is not None:
        done = ck.get("epoch", -1)
        total = (ck.get("train_args") or {}).get("epochs", 0)
        if done == -1 or (total and done + 1 >= total):
            sync_run(local, remote)
            print(f"Run '{args.name}' is already finished. best.pt: {remote / 'weights' / 'best.pt'}")
            return
        print(f"Resuming '{args.name}' after epoch {done + 1}")
        try:
            YOLO(str(last)).train(resume=True, device=device, batch=batch, data=str(args.data.resolve()),
                                  save_dir=str(local))
        except AssertionError as e:              # Ultralytics: "... is finished, nothing to resume"
            if "finished" not in str(e):
                raise
            print(f"Run '{args.name}' is already finished.")
    else:
        print(f"Starting a new run '{args.name}' from {args.model}")
        YOLO(args.model).train(
            data=str(args.data), epochs=args.epochs, imgsz=args.imgsz, batch=batch, device=device,
            project=str(args.local), name=args.name, exist_ok=True, patience=args.patience,
            cos_lr=True, close_mosaic=15, fliplr=0.0, flipud=0.0, degrees=0.0,
            hsv_h=0.01, hsv_s=0.3, hsv_v=0.3, scale=0.5, translate=0.1, mosaic=1.0,
            workers=args.workers, plots=True, seed=42, save_period=-1, cache=False,
        )
    stop.set()
    sync_run(local, remote)
    print(f"Training finished. best.pt: {remote / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
