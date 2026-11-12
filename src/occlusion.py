"""Occlusion models for Kinect-v2 25-joint skeletons in camera coordinates (x right, y up, z depth, meters).

Every function returns a boolean visibility mask vis (T,V): True = joint observed.
`apply` then turns (raw sequence, mask) into the corrupted model input.
"""
import numpy as np

V = 25
LOWER = [12, 13, 14, 15, 16, 17, 18, 19]          # hips, knees, ankles, feet
L_ARM = [4, 5, 6, 7, 21, 22]
R_ARM = [8, 9, 10, 11, 23, 24]
TORSO = [0, 1, 2, 20, 4, 8, 12, 16]                # joints that define the torso itself
HEAD = [2, 3]
FEET = [14, 15, 18, 19]
HIPS = [12, 16]
CENTER = 20                                        # spine-shoulder: upper body, rarely occluded


# ----------------------------------------------------------------- baselines / unseen test types
def random_joints(x, rng, p=0.3):
    return rng.random((x.shape[0], V)) >= p


def lower_body(x, rng=None):
    vis = np.ones((x.shape[0], V), bool)
    vis[:, LOWER] = False
    return vis


def one_arm(x, rng):
    vis = np.ones((x.shape[0], V), bool)
    vis[:, L_ARM if rng.random() < 0.5 else R_ARM] = False
    return vis


def temporal_block(x, rng, frac=0.4):
    T = x.shape[0]
    L = int(round(frac * T))
    s = rng.integers(0, T - L + 1)
    vis = np.ones((T, V), bool)
    vis[s:s + L] = False
    return vis


# ----------------------------------------------------------------- geometric (proposed) models
def furniture(x, rng, top_rel=(0.3, 1.2), width=(0.35, 1.2), gap=(0.25, 0.9), window_p=0.0):
    """A box-shaped occluder (desk/sofa/counter) standing on the floor between camera and person.

    Its front face is the plane z = z_o. A joint is hidden if the camera ray to it crosses that
    face inside the occluder's extent, i.e. the ray hits the box. Top height is relative to the
    person's foot->hip height so it ranges from "only feet" to "above the hips".
    Since the box is fixed in space, the mask changes per frame as the person moves.
    """
    T = x.shape[0]
    foot_y = np.median(x[:, FEET, 1].min(axis=1))
    hip_y = np.median(x[:, HIPS, 1].mean(axis=1))
    leg = max(hip_y - foot_y, 0.3)
    y_top = foot_y + rng.uniform(*top_rel) * leg
    z_person = np.median(x[:, TORSO, 2])
    z_o = max(z_person - rng.uniform(*gap), 0.5)
    x_c = np.median(x[:, CENTER, 0]) + rng.uniform(-0.35, 0.35)
    half_w = rng.uniform(*width) / 2

    z = np.maximum(x[..., 2], 1e-3)
    s = z_o / z                                     # ray/plane intersection scale
    xi, yi = x[..., 0] * s, x[..., 1] * s
    hidden = (z > z_o) & (yi < y_top) & (np.abs(xi - x_c) < half_w)
    if window_p and rng.random() < window_p:        # occluder only present for part of the clip
        L = rng.integers(T // 3, T + 1)
        s0 = rng.integers(0, T - L + 1)
        keep = np.zeros(T, bool)
        keep[s0:s0 + L] = True
        hidden &= keep[:, None]
    return ~hidden


def self_occlusion(x, rng=None, thickness=0.12, margin=0.04):
    """Joints hidden behind the person's own torso, seen from the camera.

    Torso = quadrilateral (L/R shoulder, R/L hip) in projected image coordinates, widened by the
    body's half-thickness. A non-torso joint is hidden if it projects inside it and is deeper
    than the torso surface.
    """
    T = x.shape[0]
    z = np.maximum(x[..., 2], 1e-3)
    u, v = x[..., 0] / z, x[..., 1] / z            # normalized image coordinates
    quad_idx = [4, 8, 16, 12]
    vis = np.ones((T, V), bool)
    cand = [j for j in range(V) if j not in TORSO and j not in HEAD]
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
        vis[t, np.array(cand)[inside & behind]] = False
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
def apply(x, vis, fill="zero", rng=None, jitter=0.05):
    """x: raw (T,V,3). Returns normalized (T,V,3) model input.

    fill='zero'  : hidden joints are missing (0 after centering) -- the common simulation.
    fill='noisy' : hidden joints look like a sensor's guess: last observed position (or nearest
                   observed frame, else parent-ish fallback) plus Gaussian jitter. Closer to what
                   Kinect / 2D pose estimators output under occlusion.
    Centering uses only *observed* data (spine-shoulder of the first frame where it is visible),
    so occlusion is never "undone" by normalization.
    """
    x = x.copy()
    T = x.shape[0]
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
                    base = x[t, CENTER] + np.array([0.0, -0.4, 0.0]) * (j in LOWER)
                x[t, j] = base + rng.normal(0, jitter, 3)
    rows = np.where(vis[:, CENTER])[0]
    if len(rows):
        c = x[rows[0], CENTER]
    else:
        any_vis = vis.any(1)
        t0 = np.where(any_vis)[0][0] if any_vis.any() else 0
        c = x[t0][vis[t0]].mean(0) if vis[t0].any() else np.zeros(3, np.float32)
    x = x - c
    if fill == "zero":
        x[~vis] = 0.0
    return x.astype(np.float32)


# ----------------------------------------------------------------- training-time augmentations
def train_aug(kind, x, rng, p=0.6):
    """Return (vis, fill) for one training sample."""
    full = np.ones((x.shape[0], V), bool)
    if kind == "none" or rng.random() > p:
        return full, "zero"
    if kind == "random":
        return random_joints(x, rng, p=rng.uniform(0.1, 0.4)), "zero"
    if kind == "lowerbody":                                  # the original proposal
        return lower_body(x), "zero"
    if kind == "geometric":                                  # ours
        r = rng.random()
        if r < 0.45:
            vis = furniture(x, rng, window_p=0.3)
        elif r < 0.7:
            vis = self_occlusion(x)
        else:
            vis = furniture(x, rng, window_p=0.3) & self_occlusion(x)
        return vis, ("noisy" if rng.random() < 0.5 else "zero")
    raise ValueError(kind)


# ----------------------------------------------------------------- test protocols
TESTS = {
    # name: (mask fn, fill)
    "clean":          (lambda x, r: np.ones((x.shape[0], V), bool), "zero"),
    "lowerbody":      (lambda x, r: lower_body(x), "zero"),
    "furniture":      (lambda x, r: furniture(x, r), "zero"),
    "furniture_noisy": (lambda x, r: furniture(x, r), "noisy"),
    "self_occ":       (lambda x, r: self_occlusion(x), "zero"),
    # unseen by every training scheme's design:
    "one_arm":        (one_arm, "zero"),
    "temporal_block": (temporal_block, "zero"),
    "random_30":      (lambda x, r: random_joints(x, r, 0.3), "zero"),
}
