"""Evaluate trained checkpoints on *real* NTU occlusion (Kinect tracking state, see ntu_raw.py).

    python -m src.eval_natural            -> results/natural.json, results/natural_table.tex

Two views of the same real clips:
  kinect  : coordinates exactly as the sensor reported them (inferred joints = sensor guesses)
  dropped : joints the sensor did not track (state < 2) are removed (zeroed), like a confidence filter
Reported on all test clips and on the heavily occluded subset (>= HEAVY of joints untracked).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import occlusion as occ
from .models import MODELS
from .train import evaluate

ROOT = Path(__file__).resolve().parents[1]
HEAVY = 0.20
ORDER = [("bilstm", "none"), ("bilstm", "geometric"), ("cnn1d", "none"), ("cnn1d", "geometric"),
         ("stgcn", "none"), ("stgcn", "random"), ("stgcn", "lowerbody"), ("stgcn", "geometric")]
NAME = {"bilstm": "Bi-LSTM", "cnn1d": "1D-CNN", "stgcn": "ST-GCN"}
AUG = {"none": "--", "random": "rand. joint", "lowerbody": "lower-body", "geometric": "geometric (ours)"}


def make_inputs(X, track):
    full = np.ones(track.shape, bool)
    kin = np.stack([occ.apply(x, v, "zero") for x, v in zip(X, full)])
    drop = np.stack([occ.apply(x, v, "zero") for x, v in zip(X, track == 2)])
    f = lambda a: a.transpose(0, 3, 1, 2).astype(np.float32)
    return f(kin), f(drop)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/ntu_dom_natural.npz"))
    ap.add_argument("--ckpt_dir", default=str(ROOT / "checkpoints"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    a = ap.parse_args()

    d = np.load(a.data)
    X, track, y = d["X"], d["track"], d["y"]
    untracked = (track < 2).mean(axis=(1, 2))
    heavy = untracked >= HEAVY
    kin, drop = make_inputs(X, track)
    print(f"{len(X)} real test clips, {100 * untracked.mean():.1f}% joints untracked on average; "
          f"{heavy.sum()} heavily occluded (>= {100 * HEAVY:.0f}%)")

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    nc = 22
    res = {"n_all": int(len(X)), "n_heavy": int(heavy.sum()),
           "untracked_mean": float(untracked.mean()), "untracked_heavy": float(untracked[heavy].mean()) if heavy.any() else 0.0,
           "models": {}}
    rows = []
    for m, aug in ORDER:
        ck = Path(a.ckpt_dir) / f"{m}_{aug}.pt"
        if not ck.exists():
            print("missing", ck)
            continue
        model = MODELS[m](nc).to(device)
        model.load_state_dict(torch.load(ck, map_location=device))
        r = {}
        for view, Xin in [("kinect", kin), ("dropped", drop)]:
            r[f"{view}_all"] = evaluate(model, Xin, y, device, nc)[0]
            r[f"{view}_heavy"] = evaluate(model, Xin[heavy], y[heavy], device, nc)[0] if heavy.any() else float("nan")
        res["models"][f"{m}_{aug}"] = r
        rows.append((m, aug, r))
        print(f"{NAME[m]:8s} {aug:10s} " + "  ".join(f"{k} {100 * v:5.1f}" for k, v in r.items()))

    out = Path(a.out)
    out.mkdir(exist_ok=True)
    (out / "natural.json").write_text(json.dumps(res, indent=2))
    cols = ["kinect_all", "kinect_heavy", "dropped_all", "dropped_heavy"]
    best = {c: max(r[c] for _, _, r in rows) for c in cols}
    L = [r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
         r" & & \multicolumn{2}{c}{Kinect output} & \multicolumn{2}{c}{Untracked dropped} \\",
         r"\cmidrule(lr){3-4}\cmidrule(l){5-6}",
         f"Model & Train aug. & All & Heavy & All & Heavy \\\\", r"\midrule"]
    for m, aug, r in rows:
        cells = [(r"\textbf{%.1f}" if abs(r[c] - best[c]) < 1e-9 else "%.1f") % (100 * r[c]) for c in cols]
        L.append(f"{NAME[m]} & {AUG[aug]} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (out / "natural_table.tex").write_text("\n".join(L) + "\n")
    print("wrote", out / "natural.json", "and natural_table.tex")


if __name__ == "__main__":
    main()
