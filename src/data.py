"""Load the pyskl/MMAction2 NTU-60 3D skeleton pickle safely and build a domestic-action subset.

Output cache: data/ntu_dom_xsub.npz with raw Kinect camera-space coordinates (meters),
resampled to T frames:  X_train (N,T,25,3), y_train, X_val, y_val.
"""
import argparse
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# NTU-60 action ids (1-indexed, as in the dataset paper) used as the "domestic" subset.
DOMESTIC = {
    1: "drink water", 2: "eat meal", 3: "brush teeth", 4: "brush hair", 5: "drop",
    6: "pick up", 8: "sit down", 9: "stand up", 11: "reading", 12: "writing",
    14: "put on jacket", 15: "take off jacket", 16: "put on shoe", 17: "take off shoe",
    28: "phone call", 29: "play with phone", 30: "type on keyboard", 37: "wipe face",
    41: "sneeze/cough", 42: "staggering", 43: "falling down", 46: "back pain",
}
CLASS_IDS = sorted(DOMESTIC)                    # 1-indexed NTU ids
NTU2LOCAL = {a - 1: i for i, a in enumerate(CLASS_IDS)}   # pkl labels are 0-indexed


class SafeUnpickler(pickle.Unpickler):
    """Only allow the numpy types needed to rebuild arrays -- refuse anything else."""
    ALLOWED = {
        ("numpy", "ndarray"), ("numpy", "dtype"),
        ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"),
        ("builtins", "slice"),
        ("_codecs", "encode"),          # used by numpy pickles to rebuild byte buffers (str -> bytes only)
    }

    def find_class(self, module, name):
        if (module, name) in self.ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")


def resample(seq, T):
    """Uniformly resample (t,V,C) to T frames."""
    t = seq.shape[0]
    idx = np.linspace(0, t - 1, T).round().astype(int)
    return seq[idx]


def build(pkl_path, T=64, split="xsub"):
    with open(pkl_path, "rb") as f:
        d = SafeUnpickler(f).load()
    train_names = set(d["split"][f"{split}_train"])
    val_names = set(d["split"][f"{split}_val"])
    out = {"train": ([], []), "val": ([], [])}
    for a in d["annotations"]:
        if a["label"] not in NTU2LOCAL:
            continue
        kp = np.asarray(a["keypoint"], dtype=np.float32)   # (M,T,25,3)
        seq = kp[0]
        valid = np.abs(seq).sum(axis=(1, 2)) > 0            # drop all-zero frames
        if valid.sum() < 2:
            continue
        seq = resample(seq[valid], T)
        part = "train" if a["frame_dir"] in train_names else "val" if a["frame_dir"] in val_names else None
        if part:
            out[part][0].append(seq)
            out[part][1].append(NTU2LOCAL[a["label"]])
    res = {}
    for part, (xs, ys) in out.items():
        res[f"X_{part}"] = np.stack(xs)
        res[f"y_{part}"] = np.asarray(ys, dtype=np.int64)
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pkl", default=str(ROOT / "data/raw/ntu60_3danno.pkl"))
    p.add_argument("--T", type=int, default=64)
    p.add_argument("--split", default="xsub")
    args = p.parse_args()
    res = build(args.pkl, args.T, args.split)
    out = ROOT / f"data/ntu_dom_{args.split}.npz"
    np.savez_compressed(out, **res)
    for k, v in res.items():
        print(k, v.shape)
    print("saved", out)
