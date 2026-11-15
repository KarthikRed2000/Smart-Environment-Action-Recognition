"""Occlusion models for 3D skeletons in camera coordinates (y up, z = depth away from camera, meters).

Works for any joint layout described by a `Layout` (NTU/Kinect-v2 25 joints, Toyota/LCR-Net 13).
Every mask function returns a boolean visibility mask vis (T,V): True = joint observed.
`apply` then turns (raw sequence, mask) into the normalized, corrupted model input.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Layout:
    name: str
    V: int
    lower: tuple     # legs incl. hips (lower-body dropout)
    l_arm: tuple
    r_arm: tuple
    torso: tuple     # joints that form the torso (never self-occluded, used for depth)
    head: tuple
    feet: tuple
    hips: tuple
    center: tuple    # normalization center (mean of these joints)
    quad: tuple      # torso outline: L shoulder, R shoulder, R hip, L hip


NTU = Layout("ntu25", 25,
             lower=(12, 13, 14, 15, 16, 17, 18, 19), l_arm=(4, 5, 6, 7, 21, 22), r_arm=(8, 9, 10, 11, 23, 24),
             torso=(0, 1, 2, 20, 4, 8, 12, 16), head=(2, 3), feet=(14, 15, 18, 19), hips=(12, 16),
             center=(20,), quad=(4, 8, 16, 12))
# LCR-Net 13 joints: 0 RAnk 1 LAnk 2 RKnee 3 LKnee 4 RHip 5 LHip 6 RWri 7 LWri 8 RElb 9 LElb 10 RSho 11 LSho 12 Head
J13 = Layout("j13", 13,
             lower=(0, 1, 2, 3, 4, 5), l_arm=(7, 9, 11), r_arm=(6, 8, 10),
             torso=(4, 5, 10, 11), head=(12,), feet=(0, 1), hips=(4, 5),
             center=(10, 11), quad=(11, 10, 4, 5))
# NTU joint index for each of the 13 shared joints (same order as J13)
NTU_TO_J13 = [18, 14, 17, 13, 16, 12, 10, 6, 9, 5, 8, 4, 3]
LAYOUTS = {"ntu25": NTU, "j13": J13}

# backwards-compatible NTU constants
V = NTU.V
LOWER, L_ARM, R_ARM = list(NTU.lower), list(NTU.l_arm), list(NTU.r_arm)
TORSO, HEAD, FEET, HIPS = list(NTU.torso), list(NTU.head), list(NTU.feet), list(NTU.hips)
CENTER = NTU.center[0]


# ----------------------------------------------------------------- baselines / unseen test types
def random_joints(x, rng, p=0.3, L=NTU):
    return rng.random((x.shape[0], L.V)) >= p


def lower_body(x, rng=None, L=NTU):
    vis = np.ones((x.shape[0], L.V), bool)
    vis[:, list(L.lower)] = False
    return vis


def one_arm(x, rng, L=NTU):
    vis = np.ones((x.shape[0], L.V), bool)
    vis[:, list(L.l_arm if rng.random() < 0.5 else L.r_arm)] = False
    return vis


def temporal_block(x, rng, frac=0.4, L=NTU):
    T = x.shape[0]
    n = int(round(frac * T))
    s = rng.integers(0, T - n + 1)
    vis = np.ones((T, L.V), bool)
    vis[s:s + n] = False
    return vis


# ----------------------------------------------------------------- geometric (proposed) models
def furniture(x, rng, top_rel=(0.3, 1.2), width=(0.35, 1.2), gap=(0.25, 0.9), window_p=0.0, L=NTU):
    """A box-shaped occluder (desk/sofa/counter) standing on the floor between camera and person.

    Its front face is the plane z = z_o. A joint is hidden if the camera ray to it crosses that
    face inside the occluder's extent, i.e. the ray hits the box. Top height is relative to the
    person's foot->hip height so it ranges from "only feet" to "above the hips".
    Since the box is fixed in space, the mask changes per frame as the person moves.
    """
    T = x.shape[0]
    foot_y = np.median(x[:, list(L.feet), 1].min(axis=1))
    hip_y = np.median(x[:, list(L.hips), 1].mean(axis=1))
    leg = max(hip_y - foot_y, 0.3)
    y_top = foot_y + rng.uniform(*top_rel) * leg
    z_person = np.median(x[:, list(L.torso), 2])
    z_o = max(z_person - rng.uniform(*gap), 0.5)
    x_c = np.median(x[:, list(L.center), 0]) + rng.uniform(-0.35, 0.35)
    half_w = rng.uniform(*width) / 2

    z = np.maximum(x[..., 2], 1e-3)
    s = z_o / z                                     # ray/plane intersection scale
    xi, yi = x[..., 0] * s, x[..., 1] * s
    hidden = (z > z_o) & (yi < y_top) & (np.abs(xi - x_c) < half_w)
    if window_p and rng.random() < window_p:        # occluder only present for part of the clip
        n = rng.integers(T // 3, T + 1)
        s0 = rng.integers(0, T - n + 1)
        keep = np.zeros(T, bool)
        keep[s0:s0 + n] = True
        hidden &= keep[:, None]
    return ~hidden


def self_occlusion(x, rng=None, thickness=0.12, margin=0.04, L=NTU):
    """Joints hidden behind the person's own torso, seen from the camera.

    Torso = quadrilateral (L/R shoulder, R/L hip) in projected image coordinates, widened by the
    body's half-thickness. A non-torso joint is hidden if it projects inside it and is deeper
    than the torso surface.
    """
    T = x.shape[0]
    z = np.maximum(x[..., 2], 1e-3)
    u, v = x[..., 0] / z, x[..., 1] / z            # normalized image coordinates
    quad_idx = list(L.quad)
    vis = np.ones((T, L.V), bool)
    cand = np.array([j for j in range(L.V) if j not in L.torso and j not in L.head])
    for t in range(T):
        zt = x[t, quad_idx, 2].mean()
        if zt < 0.3:                                # torso not tracked in this frame (zeros)
            continue
        qu, qv = u[t, quad_idx], v[t, quad_idx]
        cu = qu.mean()
        pad = thickness / zt
        qu = cu + (qu - cu) + np.sign(qu - cu) * pad
        poly = np.stack([qu, qv], 1)
        inside = _in_poly(u[t, cand], v[t, cand], poly)
        behind = x[t, cand, 2] > zt + margin
        vis[t, cand[inside & behind]] = False
    return vis


def _in_poly(px, py, poly):
    """Vectorized even-odd point-in-polygon test."""
    inside = np.zeros(px.shape, bool)
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cond = (y1 > py) != (y2 > py)
        xint = (x2 - x1) * (py - y1) / (y2 - y1 + 1e-9) + x1
        inside ^= cond & (px < xint)
    return inside


# ----------------------------------------------------------------- turning a mask into model input
def apply(x, vis, fill="zero", rng=None, jitter=0.05, L=NTU):
    """x: raw (T,V,3). Returns normalized (T,V,3) model input.

    fill='zero'  : hidden joints are missing (0 after centering) -- the common simulation.
    fill='noisy' : hidden joints look like a sensor's guess: last observed position (or nearest
                   observed frame, else a guess near the torso) plus Gaussian jitter. Closer to
                   what Kinect / 2D pose estimators output under occlusion.
    Centering uses only *observed* data (center joints of the first frame where they are all
    visible), so occlusion is never "undone" by normalization.
    """
    x = x.copy()
    ctr = list(L.center)
    if fill == "noisy" and (~vis).any():
        rng = rng or np.random.default_rng()
        for j in np.where((~vis).any(0))[0]:
            seen = np.where(vis[:, j])[0]
            for t in np.where(~vis[:, j])[0]:
                if len(seen):
                    prev = seen[seen < t]
                    src = prev[-1] if len(prev) else seen[0]
                    base = x[src, j]
                else:                                   # never seen: guess near the visible torso
                    base = x[t, ctr].mean(0) + np.array([0.0, -0.4, 0.0]) * (j in L.lower)
                x[t, j] = base + rng.normal(0, jitter, 3)
    rows = np.where(vis[:, ctr].all(1))[0]
    part = np.where(vis[:, ctr].any(1))[0]
    if len(rows):
        c = x[rows[0], ctr].mean(0)
    elif len(part):                     # e.g. one shoulder hidden (13-joint layout): use the visible one(s)
        t = part[0]
        c = x[t, [j for j in ctr if vis[t, j]]].mean(0)
    else:
        any_vis = vis.any(1)
        t0 = np.where(any_vis)[0][0] if any_vis.any() else 0
        c = x[t0][vis[t0]].mean(0) if vis[t0].any() else np.zeros(3, np.float32)
    x = x - c
    if fill == "zero":
        x[~vis] = 0.0
    return x.astype(np.float32)


# ----------------------------------------------------------------- training-time augmentations
def train_aug(kind, x, rng, p=0.6, L=NTU):
    """Return (vis, fill) for one training sample."""
    full = np.ones((x.shape[0], L.V), bool)
    if kind == "none" or rng.random() > p:
        return full, "zero"
    if kind == "random":
        return random_joints(x, rng, p=rng.uniform(0.1, 0.4), L=L), "zero"
    if kind == "lowerbody":                                  # the original proposal
        return lower_body(x, L=L), "zero"
    if kind == "geometric":                                  # ours
        r = rng.random()
        if r < 0.45:
            vis = furniture(x, rng, window_p=0.3, L=L)
        elif r < 0.7:
            vis = self_occlusion(x, L=L)
        else:
            vis = furniture(x, rng, window_p=0.3, L=L) & self_occlusion(x, L=L)
        return vis, ("noisy" if rng.random() < 0.5 else "zero")
    raise ValueError(kind)


# ----------------------------------------------------------------- test protocols
def tests(L=NTU):
    """name -> (mask fn(x, rng), fill)."""
    return {
        "clean":           (lambda x, r: np.ones((x.shape[0], L.V), bool), "zero"),
        "lowerbody":       (lambda x, r: lower_body(x, L=L), "zero"),
        "furniture":       (lambda x, r: furniture(x, r, L=L), "zero"),
        "furniture_noisy": (lambda x, r: furniture(x, r, L=L), "noisy"),
        "self_occ":        (lambda x, r: self_occlusion(x, L=L), "zero"),
        # unseen by every training scheme's design:
        "one_arm":         (lambda x, r: one_arm(x, r, L=L), "zero"),
        "temporal_block":  (lambda x, r: temporal_block(x, r, L=L), "zero"),
        "random_30":       (lambda x, r: random_joints(x, r, 0.3, L=L), "zero"),
    }


TESTS = tests(NTU)
