"""Train the junction classifier: hop arc / dot (connected) / no dot.

    python scripts/prepare_junction_data.py
    python scripts/train_junction.py

A tiny CNN (about 60k parameters) on the 32 x 32 patches from
prepare_junction_data.py; trains in a minute or two on a CPU. Saved to
circuit2dsn/models/junction.pt and used by wires.build_graph to decide whether
a 4-way crossing is a connection (dot) or two wires passing each other.
Reports accuracy per arm count against the old thickness rule.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from circuit2dsn.junction_model import JunctionNet  # noqa: E402


def augment(x: torch.Tensor) -> torch.Tensor:
    """x: (B,1,32,32) in 0..1. Dihedral flips + contrast / brightness / noise / blur."""
    B = x.shape[0]
    k = np.random.randint(0, 4)
    x = torch.rot90(x, k, (2, 3))
    if np.random.rand() < 0.5:
        x = torch.flip(x, (3,))
    gain = 0.75 + 0.5 * torch.rand(B, 1, 1, 1)
    bias = 0.08 * (torch.rand(B, 1, 1, 1) - 0.5)
    x = x * gain + bias + 0.03 * torch.randn_like(x)
    if np.random.rand() < 0.4:                     # blurrier page
        w = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3) / 16
        x = F.conv2d(x, w, padding=1)
    return x.clamp(0, 1)


def evaluate(model, d, device="cpu"):
    model.eval()
    X = torch.from_numpy(d["patches"]).float().div(255).unsqueeze(1)
    with torch.no_grad():
        p = torch.cat([F.softmax(model(X[i:i + 512]), 1) for i in range(0, len(X), 512)])
    return p.numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "junction")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--out", type=Path, default=ROOT / "circuit2dsn" / "models" / "junction.pt")
    args = ap.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    tr, va = np.load(args.data / "train.npz"), np.load(args.data / "val.npz")
    Xtr = torch.from_numpy(tr["patches"]).float().div(255).unsqueeze(1)
    ytr = torch.from_numpy(tr["label"]).long()
    model = JunctionNet()
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters, {len(Xtr)} training patches")
    counts = torch.bincount(ytr, minlength=3).float()
    weight = (counts.sum() / counts / 3).clamp(max=3.0)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=4e-3, total_steps=args.epochs * ((len(Xtr) + 127) // 128))
    best = (0.0, None)
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr))
        tot = 0.0
        for i in range(0, len(Xtr), 128):
            idx = perm[i:i + 128]
            xb, yb = augment(Xtr[idx]), ytr[idx]
            loss = F.cross_entropy(model(xb), yb, weight=weight, label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss) * len(idx)
        if ep % 5 == 0 or ep == args.epochs:
            p = evaluate(model, va)
            pred = p.argmax(1)
            acc = float((pred == va["label"]).mean())
            dot_ok = float(((p[:, 1] >= 0.5) == (va["label"] == 1)).mean())
            print(f"epoch {ep:>2}  loss {tot / len(Xtr):.3f}  val 3-class {acc:.1%}  dot-vs-not {dot_ok:.1%}  ({time.time() - t0:.0f}s)",
                  flush=True)
            if dot_ok > best[0]:
                best = (dot_ok, {k: v.clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[1])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "patch": 32}, args.out)
    p = evaluate(model, va)
    y, arms, old = va["label"], va["arms"], va["old_dot"]
    new = p[:, 1] >= 0.5
    print("\nvalidation, dot decision (new classifier vs the old thickness rule):")
    for name, m in (("all", np.ones_like(y, bool)), ("3 arms", arms == 3), ("4 arms", arms == 4), ("5+ arms", arms >= 5)):
        isdot, nodot = (y == 1) & m, (y != 1) & m
        print(f"  {name:<8} dots caught: old {(old[isdot] == 1).mean():6.1%} -> new {new[isdot].mean():6.1%}   "
              f"wrongly dot: old {(old[nodot] == 1).mean():6.1%} -> new {new[nodot].mean():6.1%}   "
              f"accuracy: old {((old[m] == 1) == (y[m] == 1)).mean():6.1%} -> new {(new[m] == (y[m] == 1)).mean():6.1%}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
