"""Train one (model, augmentation) configuration and evaluate it on every occlusion test protocol.

Example:  python -m src.train --model stgcn --aug geometric --epochs 30
Writes results/<model>_<aug>.json and checkpoints/<model>_<aug>.pt
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from . import occlusion as occ
from .models import MODELS

ROOT = Path(__file__).resolve().parents[1]


class TrainSet(Dataset):
    def __init__(self, X, y, aug, seed=0):
        self.X, self.y, self.aug = X, y, aug
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i]
        vis, fill = occ.train_aug(self.aug, x, self.rng)
        x = occ.apply(x, vis, fill, self.rng)
        # light geometric augmentation: random rotation about the vertical axis + scale
        th = self.rng.uniform(-np.pi / 12, np.pi / 12)
        R = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]], np.float32)
        x = (x @ R.T) * self.rng.uniform(0.9, 1.1)
        x[~vis] = 0.0 if fill == "zero" else x[~vis]
        return torch.from_numpy(x.transpose(2, 0, 1).copy()), self.y[i]


def seed_worker(worker_id):
    """Give each DataLoader worker its own augmentation stream."""
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(torch.initial_seed() % 2**32)


def build_test_sets(X, seed=123):
    """Fixed, reproducible corrupted copies of the validation set (cached on disk)."""
    cache = ROOT / "data/test_sets.npz"      # delete this file if occlusion.py changes
    if cache.exists():
        d = np.load(cache)
        return {k: d[k] for k in d.files if not k.startswith("visfrac_")}, \
               {k[8:]: float(d[k]) for k in d.files if k.startswith("visfrac_")}
    sets, frac = {}, {}
    for name, (fn, fill) in occ.TESTS.items():
        rng = np.random.default_rng(seed)
        xs, hidden = [], []
        for x in X:
            vis = fn(x, rng)
            xs.append(occ.apply(x, vis, fill, rng))
            hidden.append(1 - vis.mean())
        sets[name] = np.stack(xs).transpose(0, 3, 1, 2).astype(np.float32)
        frac[name] = float(np.mean(hidden))
        print(f"  test set {name:16s} hidden joints: {100 * frac[name]:.1f}%")
    np.savez_compressed(cache, **sets, **{f"visfrac_{k}": v for k, v in frac.items()})
    return sets, frac


@torch.no_grad()
def evaluate(model, X, y, device, nc, bs=256):
    model.eval()
    preds = []
    for i in range(0, len(X), bs):
        preds.append(model(torch.from_numpy(X[i:i + bs]).to(device)).argmax(1).cpu())
    p = torch.cat(preds).numpy()
    acc = float((p == y).mean())
    f1s = []
    for c in range(nc):
        tp = ((p == c) & (y == c)).sum()
        fp = ((p == c) & (y != c)).sum()
        fn = ((p != c) & (y == c)).sum()
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    return acc, float(np.mean(f1s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="stgcn", choices=list(MODELS))
    ap.add_argument("--aug", default="none", choices=["none", "random", "lowerbody", "geometric"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default=str(ROOT / "data/ntu_dom_xsub.npz"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    d = np.load(args.data)
    Xtr, ytr, Xva, yva = d["X_train"], d["y_train"], d["X_val"], d["y_val"]
    nc = int(ytr.max()) + 1
    print(f"train {Xtr.shape} val {Xva.shape} classes {nc} device {device}")
    tests, hidden = build_test_sets(Xva)

    dl = DataLoader(TrainSet(Xtr, ytr, args.aug, args.seed), batch_size=args.bs, shuffle=True,
                    num_workers=args.workers, persistent_workers=args.workers > 0, drop_last=True,
                    worker_init_fn=seed_worker)
    model = MODELS[args.model](nc).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * len(dl), pct_start=0.1)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)

    tag = f"{args.model}_{args.aug}"
    for ep in range(args.epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            loss = lossf(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(yb)
            n += len(yb)
        msg = f"[{tag}] ep {ep + 1}/{args.epochs} loss {tot / n:.3f} ({time.time() - t0:.0f}s)"
        if (ep + 1) % 5 == 0 or ep == args.epochs - 1:
            acc, _ = evaluate(model, tests["clean"], yva, device, nc)
            msg += f" clean acc {100 * acc:.1f}"
        print(msg, flush=True)

    # latency (batch 1, after warm-up)
    model.eval()
    x1 = torch.from_numpy(tests["clean"][:1]).to(device)
    with torch.no_grad():
        for _ in range(10):
            model(x1)
        sync = torch.cuda.synchronize if device.type == "cuda" else \
            torch.mps.synchronize if device.type == "mps" else (lambda: None)
        sync()
        t0 = time.time()
        for _ in range(50):
            model(x1)
        sync()
    lat_ms = (time.time() - t0) / 50 * 1000

    res = {"model": args.model, "aug": args.aug, "epochs": args.epochs, "params": nparam,
           "latency_ms_bs1": lat_ms, "hidden_frac": hidden, "tests": {}}
    for name, X in tests.items():
        acc, f1 = evaluate(model, X, yva, device, nc)
        res["tests"][name] = {"acc": acc, "f1": f1}
        print(f"  {name:16s} acc {100 * acc:5.1f}  F1 {100 * f1:5.1f}")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "checkpoints").mkdir(exist_ok=True)
    (ROOT / f"results/{tag}.json").write_text(json.dumps(res, indent=2))
    torch.save(model.state_dict(), ROOT / f"checkpoints/{tag}.pt")


if __name__ == "__main__":
    main()
