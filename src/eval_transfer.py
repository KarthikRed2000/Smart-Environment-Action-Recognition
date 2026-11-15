"""Zero-shot transfer: NTU-trained (13 shared joints) ST-GCN -> Toyota Smarthome real homes.

    python -m src.eval_transfer        -> results/transfer.json, results/transfer_table.tex

Models are trained on the 22 NTU classes (python -m src.train --dataset ntu13 ...); at test time
the logits are restricted to the 8 NTU classes that have a Toyota counterpart (see toyota.TO_NTU),
and every Toyota clip of those classes is used (no Toyota data is ever trained on).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import occlusion as occ
from .data import CLASS_IDS, DOMESTIC
from .models import MODELS
from .toyota import TO_NTU

ROOT = Path(__file__).resolve().parents[1]
AUGS = ["none", "random", "lowerbody", "geometric"]
AUG = {"none": "--", "random": "rand. joint", "lowerbody": "lower-body (prop.)", "geometric": "geometric (ours)"}


def toyota_transfer_set(path=ROOT / "data/toyota_all.npz"):
    d = np.load(path)
    classes = list(d["classes"])
    ntu_local = {DOMESTIC[a]: i for i, a in enumerate(CLASS_IDS)}           # name -> 0..21
    targets = sorted({ntu_local[v] for v in TO_NTU.values()})
    keep, y = [], []
    for i, c in enumerate(d["y"]):
        name = classes[c]
        if name in TO_NTU:
            keep.append(i)
            y.append(ntu_local[TO_NTU[name]])
    X = np.stack([occ.apply(x, np.ones(x.shape[:2], bool), L=occ.J13) for x in d["X"][keep]])
    return X.transpose(0, 3, 1, 2).astype(np.float32), np.array(y), targets


@torch.no_grad()
def predict(model, X, targets, device, bs=256):
    out = []
    for i in range(0, len(X), bs):
        logits = model(torch.from_numpy(X[i:i + bs]).to(device))[:, targets]
        out.append(torch.tensor(targets)[logits.argmax(1).cpu()])
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default=str(ROOT / "checkpoints"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    X, y, targets = toyota_transfer_set()
    names = {t: DOMESTIC[CLASS_IDS[t]] for t in targets}
    print(f"{len(X)} Toyota clips over {len(targets)} shared classes: " +
          ", ".join(f"{names[t]} {int((y == t).sum())}" for t in targets))
    res = {"n": int(len(X)), "classes": [names[t] for t in targets], "models": {}}
    for aug in AUGS:
        ck = Path(a.ckpt_dir) / f"ntu13_stgcn_{aug}.pt"
        if not ck.exists():
            print("missing", ck)
            continue
        model = MODELS["stgcn"](22, V=13).to(device)
        model.load_state_dict(torch.load(ck, map_location=device))
        model.eval()
        p = predict(model, X, targets, device)
        acc = float((p == y).mean())
        per = {names[t]: float((p[y == t] == t).mean()) for t in targets}
        res["models"][aug] = {"acc": acc, "mean_class_acc": float(np.mean(list(per.values()))), "per_class": per}
        print(f"ST-GCN {aug:10s} top-1 {100 * acc:5.1f}  mean-class {100 * res['models'][aug]['mean_class_acc']:5.1f}")
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    (out / "transfer.json").write_text(json.dumps(res, indent=2))
    if res["models"]:
        L = [r"\begin{tabular}{@{}lcc@{}}", r"\toprule", r"Train aug. (NTU) & Top-1 & Mean class \\", r"\midrule"]
        for aug, r in res["models"].items():
            L.append(f"{AUG[aug]} & {100 * r['acc']:.1f} & {100 * r['mean_class_acc']:.1f} \\\\")
        L += [r"\bottomrule", r"\end{tabular}"]
        (out / "transfer_table.tex").write_text("\n".join(L) + "\n")
        print("wrote", out / "transfer.json")


if __name__ == "__main__":
    main()
