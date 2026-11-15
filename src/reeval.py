"""Re-evaluate saved checkpoints on the (rebuilt) corrupted test sets without retraining.

    python -m src.reeval --dataset toyota      # updates results/toyota_stgcn_*.json in place

Delete data/test_sets_<dataset>.npz first if occlusion.py changed.
"""
import argparse
import json
from pathlib import Path

import torch

from .models import MODELS
from .train import build_test_sets, evaluate, load_dataset

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ntu", "ntu13", "toyota"])
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    _, _, Xva, yva, L, cache = load_dataset(a.dataset)
    tests, hidden = build_test_sets(Xva, L=L, cache_name=cache)
    prefix = "" if a.dataset == "ntu" else f"{a.dataset}_"
    for f in sorted((ROOT / "results").glob(f"{prefix}*.json")):
        r = json.loads(f.read_text())
        if r.get("dataset", "ntu") != a.dataset or "tests" not in r:
            continue
        tag = f.stem
        nc = int(yva.max()) + 1
        model = MODELS[r["model"]](nc, V=L.V).to(device)
        model.load_state_dict(torch.load(ROOT / f"checkpoints/{tag}.pt", map_location=device))
        r["hidden_frac"] = hidden
        for name, X in tests.items():
            acc, f1, mcls = evaluate(model, X, yva, device, nc, mca=True)
            r["tests"][name] = {"acc": acc, "f1": f1, "mean_class_acc": mcls}
        f.write_text(json.dumps(r, indent=2))
        print(tag, " ".join(f"{k} {100 * v['acc']:.1f}" for k, v in r["tests"].items()))


if __name__ == "__main__":
    main()
