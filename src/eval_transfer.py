"""Zero-shot transfer: NTU-trained (13 shared joints) ST-GCN -> Toyota Smarthome real homes.

    python -m src.eval_transfer        -> results/transfer.json, results/transfer_table.tex

Models are trained on the 22 NTU classes; at test time the logits are restricted to the 8 NTU
classes that have a Toyota counterpart (see toyota.TO_NTU), and every Toyota clip of those classes
is used (no Toyota data is ever trained on). Two input protocols:
  raw         : checkpoints/ntu13_stgcn_<aug>[_sK].pt, Toyota as stored (first-frame centering)
  harmonized  : checkpoints/ntu13h_stgcn_<aug>[_sK].pt, both datasets per-frame root-centered and
                torso-scaled (occlusion.harmonize) -- removes the global-motion / scale mismatch.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import occlusion as occ
from . import seeds as S
from .data import CLASS_IDS, DOMESTIC
from .models import MODELS
from .toyota import TO_NTU

ROOT = Path(__file__).resolve().parents[1]
AUGS = ["none", "random", "lowerbody", "geometric", "combined"]
AUG = {"none": "--", "random": "rand. joint", "lowerbody": "lower-body (prop.)",
       "geometric": "geometric (ours)", "combined": "combined (ours)"}
PROTOCOLS = {"raw": "ntu13", "harmonized": "ntu13h"}


def toyota_transfer_set(path=ROOT / "data/toyota_all.npz", harmonized=False):
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
    X = d["X"][keep]
    if harmonized:
        X = occ.harmonize(X)
    X = np.stack([occ.apply(x, np.ones(x.shape[:2], bool), L=occ.J13) for x in X])
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
    res = {}
    for proto, ds in PROTOCOLS.items():
        X, y, targets = toyota_transfer_set(harmonized=proto == "harmonized")
        names = {t: DOMESTIC[CLASS_IDS[t]] for t in targets}
        res.setdefault("n", int(len(X)))
        res.setdefault("classes", [names[t] for t in targets])
        res.setdefault("class_counts", {names[t]: int((y == t).sum()) for t in targets})
        res[proto] = {}
        for aug in AUGS:
            runs = []
            for tag, ck in S.ckpts(f"{ds}_stgcn_{aug}", a.ckpt_dir):
                model = MODELS["stgcn"](22, V=13).to(device)
                model.load_state_dict(torch.load(ck, map_location=device))
                model.eval()
                p = predict(model, X, targets, device)
                per = {names[t]: float((p[y == t] == t).mean()) for t in targets}
                runs.append({"tag": tag, "acc": float((p == y).mean()),
                             "mean_class_acc": float(np.mean(list(per.values()))), "per_class": per})
            if not runs:
                continue
            accs, mcas = [r["acc"] for r in runs], [r["mean_class_acc"] for r in runs]
            res[proto][aug] = {"runs": runs, "acc": S.ms(accs)[0], "acc_std": S.ms(accs)[1],
                               "mean_class_acc": S.ms(mcas)[0], "mean_class_acc_std": S.ms(mcas)[1],
                               "per_class": {k: float(np.mean([r["per_class"][k] for r in runs])) for k in runs[0]["per_class"]}}
            print(f"[{proto:10s}] ST-GCN {aug:10s} ({len(runs)} seeds) top-1 {S.fmt(accs, std=True)}  "
                  f"mean-class {S.fmt(mcas, std=True)}")
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    (out / "transfer.json").write_text(json.dumps(res, indent=2))
    L = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
         r" & \multicolumn{2}{c}{Raw inputs} & \multicolumn{2}{c}{Harmonized} \\",
         r"\cmidrule(lr){2-3}\cmidrule(l){4-5}",
         r"Train aug. (NTU) & Top-1 & Mean cls. & Top-1 & Mean cls. \\", r"\midrule"]
    for aug in AUGS:
        cells = []
        for proto in PROTOCOLS:
            r = res[proto].get(aug)
            if r is None:
                cells += ["--", "--"]
            else:
                cells += [S.fmt([x["acc"] for x in r["runs"]]), S.fmt([x["mean_class_acc"] for x in r["runs"]])]
        if any(c != "--" for c in cells):
            L.append(f"{AUG[aug]} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (out / "transfer_table.tex").write_text("\n".join(L) + "\n")
    print("wrote", out / "transfer.json")


if __name__ == "__main__":
    main()
