"""Figures for the final report.

    python -m src.figures      -> report/fig_occlusion.pdf, report/fig_toyota.pdf

fig_occlusion: one NTU test frame in the camera image under (a) lower-body dropout (proposal),
(b) our furniture occluder, (c) our self-occlusion, (d) real Kinect tracking failures.
fig_toyota: Toyota mean-class accuracy per occlusion protocol, mean +- std over seeds.
"""
import numpy as np

from . import occlusion as occ
from . import seeds as S
from .models import _EDGES

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
E0 = 1 if min(min(e) for e in _EDGES) == 1 else 0
BONES = [(a - E0, b - E0) for a, b in _EDGES]
VIS_C, HID_C = "#2b6cb0", "#d64545"


def furniture_params(x, seed, L=occ.NTU, top_rel=(0.3, 1.2), width=(0.35, 1.2), gap=(0.25, 0.9)):
    """Re-draw the occluder parameters exactly as occlusion.furniture does for the same rng seed."""
    rng = np.random.default_rng(seed)
    foot_y = np.median(x[:, list(L.feet), 1].min(axis=1))
    hip_y = np.median(x[:, list(L.hips), 1].mean(axis=1))
    leg = max(hip_y - foot_y, 0.3)
    y_top = foot_y + rng.uniform(*top_rel) * leg
    z_person = np.median(x[:, list(L.torso), 2])
    z_o = max(z_person - rng.uniform(*gap), 0.5)
    x_c = np.median(x[:, list(L.center), 0]) + rng.uniform(-0.35, 0.35)
    half_w = rng.uniform(*width) / 2
    return y_top, z_o, x_c, half_w, foot_y


def draw(ax, xt, vis, title):
    z = np.maximum(xt[:, 2], 1e-3)
    u, v = -xt[:, 0] / z, xt[:, 1] / z             # mirror x so the image looks like the RGB view
    for a, b in BONES:
        hid = not (vis[a] and vis[b])
        ax.plot([u[a], u[b]], [v[a], v[b]], color="#bbbbbb" if hid else "#555555",
                lw=1.0 if hid else 1.6, ls=":" if hid else "-", zorder=1)
    ax.scatter(u[vis], v[vis], s=14, c=VIS_C, zorder=3)
    ax.scatter(u[~vis], v[~vis], s=22, c=HID_C, marker="x", zorder=3)
    ax.set_title(title, fontsize=8)
    ax.set_aspect("equal")
    ax.axis("off")
    return u, v


def fig_occlusion():
    d = np.load(ROOT / "data/ntu_dom_xsub.npz")
    X, y = d["X_val"], d["y_val"]
    nat = np.load(ROOT / "data/ntu_dom_natural.npz")
    rng = np.random.default_rng(0)
    # a clip where the furniture occluder hides a moderate part of the body
    best = None
    for i in rng.permutation(len(X))[:400]:
        x = X[i]
        if (np.abs(x).sum(-1) == 0).any():
            continue
        for s in range(20):
            vis = occ.furniture(x, np.random.default_rng(s))
            f = (~vis).mean()
            if 0.25 < f < 0.4 and (~vis[:, list(occ.FEET)]).all() and vis[:, 3].all():
                best = (i, s)
                break
        if best:
            break
    i, s = best
    x = X[i]
    t = 32
    # self-occlusion example: a frame with the most joints behind the torso
    so_best, so_cnt = None, 0
    for j in rng.permutation(len(X))[:1500]:
        so = occ.self_occlusion(X[j])
        c = (~so).sum(1)
        if c.max() > so_cnt:
            so_best, so_cnt = (j, int(c.argmax())), c.max()
    # real Kinect example: heavy untracked frame
    # typical heavily occluded frame: 25-35% untracked, and every bone (tracked or not) has a
    # plausible length, so the sensor's guesses are still near the body
    tr, XN = nat["track"], nat["X"]
    bones = np.array(BONES)
    k, kt, kbest = None, None, 1e9
    for c in np.where((tr < 2).mean((1, 2)) > 0.2)[0][:800]:
        for f in range(0, 64, 2):
            if not 0.25 <= (tr[c, f] < 2).mean() <= 0.35 or (np.abs(XN[c, f]).sum(-1) == 0).any():
                continue
            bl = np.linalg.norm(XN[c, f, bones[:, 0]] - XN[c, f, bones[:, 1]], axis=-1).max()
            if bl < kbest:
                k, kt, kbest = c, f, bl

    fig, axs = plt.subplots(1, 4, figsize=(7.0, 2.3))
    draw(axs[0], x[t], occ.lower_body(x)[t], "(a) lower-body dropout")
    vis = occ.furniture(x, np.random.default_rng(s))
    y_top, z_o, x_c, hw, foot_y = furniture_params(x, s)
    draw(axs[1], x[t], vis[t], "(b) furniture (ours)")
    bx = [-(x_c - hw) / z_o, -(x_c + hw) / z_o]
    by = [(foot_y - 0.1) / z_o, y_top / z_o]
    axs[1].add_patch(plt.Rectangle((min(bx), by[0]), abs(bx[1] - bx[0]), by[1] - by[0],
                                   color="#8b5e3c", alpha=0.25, zorder=2, lw=0))
    j, jt = so_best
    draw(axs[2], X[j][jt], occ.self_occlusion(X[j])[jt], "(c) self-occl. (ours), back view")
    draw(axs[3], nat["X"][k][kt], nat["track"][k][kt] == 2, "(d) real Kinect untracked")
    fig.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=VIS_C, label="visible"),
                        plt.Line2D([], [], marker="x", ls="", color=HID_C, label="hidden / untracked")],
               loc="lower center", ncol=2, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(ROOT / "report/fig_occlusion.pdf")
    print("wrote report/fig_occlusion.pdf")


def fig_toyota():
    cols = ["clean", "furniture", "self_occ", "one_arm", "temporal_block", "random_30"]
    lab = ["Real", "+Furn.", "+Self", "+Arm", "+Temp.", "+Rand."]
    augs = [("none", "no aug.", "#9e9e9e"), ("random", "rand. joint", "#e0a030"),
            ("lowerbody", "lower-body", "#7fb069"), ("geometric", "geometric", "#2b6cb0"),
            ("combined", "combined", "#173f73")]
    augs = [a for a in augs if S.load(f"toyota_stgcn_{a[0]}")]
    w = 0.8 / len(augs)
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    for k, (aug, name, c) in enumerate(augs):
        rs = S.load(f"toyota_stgcn_{aug}")
        m = [100 * S.ms([r["tests"][col]["mean_class_acc"] for r in rs])[0] for col in cols]
        e = [100 * S.ms([r["tests"][col]["mean_class_acc"] for r in rs])[1] for col in cols]
        ax.bar(np.arange(len(cols)) + (k - (len(augs) - 1) / 2) * w, m, w, yerr=e, color=c, label=name,
               error_kw={"lw": 0.6, "capsize": 1})
    ax.set_xticks(range(len(cols)), lab, fontsize=7)
    ax.set_ylabel("mean class acc. (%)", fontsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=6, ncol=len(augs), frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              handlelength=1, columnspacing=0.8)
    fig.tight_layout()
    fig.savefig(ROOT / "report/fig_toyota.pdf")
    print("wrote report/fig_toyota.pdf")


if __name__ == "__main__":
    fig_occlusion()
    fig_toyota()
