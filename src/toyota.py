"""Toyota Smarthome (Trimmed) skeletons v1.2 -> data/toyota_all.npz.

Each clip JSON: {"njts": 13, "frames": [[{"pose3d": 39 floats, "pose2d": 26 floats}], ...]}
pose3d is stored planar (all x, all y, all z), pelvis-centered, meters; x points to image-left,
y up, z = depth away from the camera (checked against pose2d). We place a virtual camera
CAM_DIST m in front of the pelvis so the geometric occlusion code (camera rays) applies.

    python -m src.toyota [--zip data/toyota_smarthome_skeleton_v1.2.zip]
"""
import argparse
import json
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CAM_DIST = 3.5
T_OUT = 64
# official cross-subject protocol (Das et al., ICCV 2019)
CS_TRAIN_SUBJECTS = {3, 4, 6, 7, 9, 12, 13, 15, 17, 19, 25}
# Toyota classes that match an NTU domestic class (for the NTU -> Toyota transfer experiment)
TO_NTU = {
    "Drink.Fromcup": "drink water", "Drink.Frombottle": "drink water", "Drink.Fromcan": "drink water",
    "Drink.Fromglass": "drink water", "Eat.Attable": "eat meal", "Eat.Snack": "eat meal",
    "Sitdown": "sit down", "Getup": "stand up", "Readbook": "reading", "Usetelephone": "phone call",
    "Uselaptop": "type on keyboard", "Usetablet": "play with phone",
}


def parse_name(n):
    # e.g. Cook.Cleandishes_p02_r00_v02_c03_pose3d.json
    stem = Path(n).name.replace("_pose3d.json", "")
    cls, p, r, v, c = stem.rsplit("_", 4)
    return stem, cls, int(p[1:]), int(c[1:])


def _load(args):
    zpath, name = args
    with zipfile.ZipFile(zpath) as z:
        d = json.loads(z.read(name))
    poses = [f[0]["pose3d"] for f in d["frames"] if f]          # first (only) person per frame
    if len(poses) < 2:
        return None
    P = np.asarray(poses, np.float32).reshape(-1, 3, 13).transpose(0, 2, 1)   # (t,13,3)
    P[..., 2] += CAM_DIST                                        # pelvis-centered -> camera depth
    idx = np.linspace(0, len(P) - 1, T_OUT).round().astype(int)
    return P[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=str(ROOT / "data/toyota_smarthome_skeleton_v1.2.zip"))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    with zipfile.ZipFile(a.zip) as z:
        names = sorted(n for n in z.namelist() if n.endswith("_pose3d.json"))
    meta = [parse_name(n) for n in names]
    with Pool(a.workers) as pool:
        seqs = pool.map(_load, [(a.zip, n) for n in names], chunksize=64)
    keep = [i for i, s in enumerate(seqs) if s is not None]
    classes = sorted({meta[i][1] for i in keep})
    cid = {c: k for k, c in enumerate(classes)}
    out = ROOT / "data/toyota_all.npz"
    np.savez_compressed(
        out, X=np.stack([seqs[i] for i in keep]),
        y=np.array([cid[meta[i][1]] for i in keep], np.int64),
        subject=np.array([meta[i][2] for i in keep]), camera=np.array([meta[i][3] for i in keep]),
        names=np.array([meta[i][0] for i in keep]), classes=np.array(classes))
    print(f"{len(keep)}/{len(names)} clips, {len(classes)} classes -> {out}")


def load_cs(path=ROOT / "data/toyota_all.npz"):
    """Cross-subject split: X_train, y_train, X_val, y_val, classes."""
    d = np.load(path)
    tr = np.isin(d["subject"], list(CS_TRAIN_SUBJECTS))
    return d["X"][tr], d["y"][tr], d["X"][~tr], d["y"][~tr], list(d["classes"])


if __name__ == "__main__":
    main()
