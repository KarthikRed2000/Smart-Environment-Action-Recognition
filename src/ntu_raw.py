"""Build a *naturally occluded* NTU test set from the official raw .skeleton files.

Each Kinect joint has a trackingState: 2 = tracked, 1 = inferred (sensor guess), 0 = not tracked.
Inferred / untracked joints are real sensor failures (mostly occlusion), so they give us
real-occlusion test data without any simulation.

Usage (after unzipping nturgbd_skeletons_s001_to_s017.zip into data/raw/nturgb+d_skeletons/):
    python -m src.ntu_raw
Writes data/ntu_dom_natural.npz with, for the cross-subject *test* clips of our 22 classes:
    X (N,T,25,3) raw camera coords, track (N,T,25) tracking state, y (N,), names (N,)
"""
import argparse
from pathlib import Path

import numpy as np

from .data import NTU2LOCAL, resample

ROOT = Path(__file__).resolve().parents[1]
# official NTU-60 cross-subject training subjects; all others are test
XSUB_TRAIN = {1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38}


def read_skeleton(path):
    """Parse one .skeleton file -> dict body_id -> (frames list, xyz (t,25,3), state (t,25))."""
    with open(path) as f:
        lines = f.read().split("\n")
    i = 0
    n_frames = int(lines[i]); i += 1
    bodies = {}
    for t in range(n_frames):
        n_bodies = int(lines[i]); i += 1
        for _ in range(n_bodies):
            body_id = lines[i].split()[0]; i += 1
            n_joints = int(lines[i]); i += 1
            xyz = np.zeros((n_joints, 3), np.float32)
            st = np.zeros(n_joints, np.int8)
            for j in range(n_joints):
                v = lines[i].split(); i += 1
                xyz[j] = [float(v[0]), float(v[1]), float(v[2])]
                st[j] = int(v[11])
            b = bodies.setdefault(body_id, ([], [], []))
            b[0].append(t); b[1].append(xyz); b[2].append(st)
    return {k: (np.array(f), np.stack(x), np.stack(s)) for k, (f, x, s) in bodies.items()}


def main_body(bodies):
    """Pick the actor: the body tracked longest, ties broken by motion (as in common NTU tooling)."""
    def score(b):
        frames, xyz, _ = b
        return (len(frames), float(xyz.std(0).sum()))
    return max(bodies.values(), key=score)


def build(skel_dir, T=64):
    X, TR, Y, N = [], [], [], []
    files = sorted(Path(skel_dir).glob("*.skeleton"))
    if not files:
        raise SystemExit(f"no .skeleton files in {skel_dir}")
    for p in files:
        name = p.stem                                  # e.g. S001C001P001R001A001
        subject, action = int(name[9:12]), int(name[17:20])
        if subject in XSUB_TRAIN or (action - 1) not in NTU2LOCAL:
            continue
        try:
            bodies = read_skeleton(p)
        except (ValueError, IndexError):
            continue                                   # a few official files are empty/corrupt
        if not bodies:
            continue
        _, xyz, st = main_body(bodies)
        if len(xyz) < 2:
            continue
        idx = np.linspace(0, len(xyz) - 1, T).round().astype(int)
        X.append(xyz[idx]); TR.append(st[idx]); Y.append(NTU2LOCAL[action - 1]); N.append(name)
    return np.stack(X), np.stack(TR), np.array(Y, np.int64), np.array(N)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skel_dir", default=str(ROOT / "data/raw/nturgb+d_skeletons"))
    ap.add_argument("--T", type=int, default=64)
    a = ap.parse_args()
    X, TR, Y, N = build(a.skel_dir, a.T)
    out = ROOT / "data/ntu_dom_natural.npz"
    np.savez_compressed(out, X=X, track=TR, y=Y, names=N)
    occ = (TR < 2).mean(axis=(1, 2))
    print(f"{len(X)} test clips; joints not tracked: mean {100 * occ.mean():.1f}%, "
          f"clips with >=10% untracked: {(occ >= 0.10).sum()}, >=20%: {(occ >= 0.20).sum()}")
    print("saved", out)
