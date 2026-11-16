"""Fast end-to-end checks (~1 min on CPU). No full training.

    python -m tests.smoke_test            # synthetic data only
    python -m tests.smoke_test --real     # also checks data/ntu_dom_xsub.npz if it exists
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import occlusion as occ            # noqa: E402
from src.models import MODELS, ntu_graph    # noqa: E402
from src.train import TrainSet, evaluate    # noqa: E402

OK = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    OK.append(bool(cond))


def fake_person(T=64, z=3.0, right_arm_behind=False, seed=0):
    """Standing person ~z m from camera, small motion. Kinect-v2 joint order."""
    rng = np.random.default_rng(seed)
    x = np.zeros((T, 25, 3), np.float32)
    x[..., 2] = z
    x[:, [0, 1, 20, 2, 3], 1] = [0.0, 0.25, 0.5, 0.6, 0.75]
    x[:, [4, 8], 0], x[:, [4, 8], 1] = [-0.2, 0.2], 0.5
    x[:, [5, 6, 7, 21, 22], 0], x[:, [5, 6, 7, 21, 22], 1] = -0.25, 0.2
    x[:, [9, 10, 11, 23, 24], 0], x[:, [9, 10, 11, 23, 24], 1] = 0.25, 0.2
    x[:, [12, 16], 0] = [-0.1, 0.1]
    x[:, [13, 14, 15], 0], x[:, [17, 18, 19], 0] = -0.1, 0.1
    x[:, [13, 17], 1], x[:, [14, 15, 18, 19], 1] = -0.45, -0.85
    if right_arm_behind:
        x[:, [9, 10, 11, 23, 24], 0] = 0.05
        x[:, [9, 10, 11, 23, 24], 2] = z + 0.3
    return x + rng.normal(0, 0.005, x.shape).astype(np.float32)


def test_graph():
    A = ntu_graph()
    check("graph: 3 partitions of 25x25", tuple(A.shape) == (3, 25, 25))
    check("graph: 24 bones", int((A[1] > 0).sum()) == 24)


def test_models():
    x = torch.randn(4, 3, 64, 25)
    for k, M in MODELS.items():
        m = M(22)
        out = m(x)
        check(f"model {k}: output (4,22)", tuple(out.shape) == (4, 22))
        out.sum().backward()
        check(f"model {k}: gradients finite", all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))


def test_occlusion_semantics():
    rng = np.random.default_rng(0)
    x = fake_person()
    vis = occ.lower_body(x)
    check("lower_body hides exactly the 8 leg joints", (~vis).all(0).sum() == 8 and vis[:, occ.LOWER].sum() == 0)

    # a tall wide occluder right in front must hide the feet but not the head
    vis = occ.furniture(x, rng, top_rel=(1.0, 1.0), width=(3.0, 3.0), gap=(0.5, 0.5))
    check("furniture: feet hidden", (~vis[:, occ.FEET]).all())
    check("furniture: head visible", vis[:, 3].all())
    # a tiny occluder far to the side hides nothing
    vis = occ.furniture(x + np.array([5.0, 0, 0], np.float32) * 0, rng, top_rel=(0.3, 0.3), width=(0.01, 0.01))
    check("furniture: tiny occluder hides little", (~vis).mean() < 0.1)

    check("self_occ: frontal person, nothing hidden", occ.self_occlusion(fake_person()).all())
    vis = occ.self_occlusion(fake_person(right_arm_behind=True))
    check("self_occ: arm behind torso is hidden", (~vis[:, [9, 10, 11]]).all())
    check("self_occ: left arm still visible", vis[:, [5, 6, 7]].all())

    # frames where Kinect lost the body (all zeros) must not crash or create NaNs
    xz = fake_person()
    xz[10:20] = 0
    vis = occ.self_occlusion(xz)
    check("self_occ: zero frames handled", np.isfinite(occ.apply(xz, vis)).all())

    tb = occ.temporal_block(x, rng, 0.4)
    check("temporal_block: ~40% frames removed", abs((~tb.all(1)).mean() - 0.4) < 0.03)


def test_apply():
    rng = np.random.default_rng(0)
    x = fake_person()
    vis = occ.lower_body(x)
    z = occ.apply(x, vis, "zero")
    check("apply zero: hidden joints are 0", np.all(z[:, occ.LOWER] == 0))
    check("apply: centered on spine-shoulder", np.allclose(z[0, occ.CENTER], 0, atol=1e-6))
    n = occ.apply(x, vis, "noisy", rng)
    check("apply noisy: hidden joints are not 0", np.abs(n[:, occ.LOWER]).sum() > 0)
    check("apply noisy: finite", np.isfinite(n).all())
    # centering must not use hidden data: hide spine-shoulder in frame 0 only
    vis2 = np.ones((64, 25), bool)
    vis2[0, occ.CENTER] = False
    c = occ.apply(x, vis2, "zero")
    check("apply: center falls back to first visible frame", np.allclose(c[1, occ.CENTER], 0, atol=1e-6))


def test_augs_and_tests():
    rng = np.random.default_rng(0)
    xs = [fake_person(seed=i) for i in range(8)]
    for kind in ["none", "random", "lowerbody", "geometric", "combined"]:
        out = [occ.apply(x, *occ.train_aug(kind, x, rng)[:1], occ.train_aug(kind, x, rng)[1], rng) for x in xs]
        check(f"train_aug {kind}: finite, shape", all(np.isfinite(o).all() and o.shape == (64, 25, 3) for o in out))
    for name, (fn, fill) in occ.TESTS.items():
        vis = fn(xs[0], rng)
        check(f"test protocol {name}: mask shape", vis.shape == (64, 25) and vis.dtype == bool)


def test_training_step(X=None, y=None):
    """Two optimizer steps + evaluation for every model x augmentation (catches wiring bugs)."""
    if X is None:
        X = np.stack([fake_person(seed=i) for i in range(32)])
        y = np.arange(32) % 4
    nc = int(y.max()) + 1
    for model_name in MODELS:
        for aug in ["none", "random", "lowerbody", "geometric"]:
            ds = TrainSet(X, y, aug)
            dl = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True)
            m = MODELS[model_name](nc)
            opt = torch.optim.AdamW(m.parameters(), 1e-3)
            losses = []
            for i, (xb, yb) in enumerate(dl):
                loss = nn.functional.cross_entropy(m(xb), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())
                if i == 1:
                    break
            Xt = np.stack([occ.apply(x, np.ones((64, 25), bool)) for x in X[:16]]).transpose(0, 3, 1, 2)
            acc, f1 = evaluate(m, Xt.astype(np.float32), y[:16], torch.device("cpu"), nc)
            check(f"train step {model_name}/{aug}: finite loss, eval ok",
                  np.isfinite(losses).all() and 0 <= acc <= 1 and 0 <= f1 <= 1)


def test_real_data():
    p = ROOT / "data/ntu_dom_xsub.npz"
    if not p.exists():
        print("SKIP real data (data/ntu_dom_xsub.npz missing -- run python -m src.data)")
        return
    d = dict(np.load(p))          # decompress once (NpzFile re-reads on every access)
    check("real: shapes", d["X_train"].shape[1:] == (64, 25, 3) and len(d["X_train"]) == len(d["y_train"]))
    check("real: 22 classes in both splits", len(np.unique(d["y_train"])) == 22 and len(np.unique(d["y_val"])) == 22)
    check("real: camera-space depth (z ~ 2-5 m)", 1.5 < np.median(d["X_val"][..., 2]) < 5.0)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(d["X_val"]), 100, replace=False)
    for name, (fn, fill) in occ.TESTS.items():
        outs, hid = [], []
        for i in idx:
            v = fn(d["X_val"][i], rng)
            outs.append(occ.apply(d["X_val"][i], v, fill, rng))
            hid.append(1 - v.mean())
        check(f"real: {name:16s} finite (hidden {100 * np.mean(hid):4.1f}%)", np.isfinite(np.stack(outs)).all())
    test_training_step(d["X_train"][:32], d["y_train"][:32] % 4)


def test_j13():
    """13-joint (Toyota / shared) layout: masks, graph, models, training step."""
    rng = np.random.default_rng(0)
    L = occ.J13
    x = fake_person()[:, occ.NTU_TO_J13]                     # NTU person mapped to 13 joints
    check("j13: mapping gives 13 joints, head highest", x.shape == (64, 13, 3) and x[0, 12, 1] == x[0, :, 1].max())
    check("j13 graph: 13 bones", int((ntu_graph(layout="j13")[1] > 0).sum()) == 13)
    vis = occ.lower_body(x, L=L)
    check("j13 lower_body hides 6 leg joints", (~vis).all(0).sum() == 6)
    vis = occ.furniture(x, rng, top_rel=(1.0, 1.0), width=(3.0, 3.0), gap=(0.5, 0.5), L=L)
    check("j13 furniture: ankles hidden, head visible", (~vis[:, list(L.feet)]).all() and vis[:, 12].all())
    xb = fake_person(right_arm_behind=True)[:, occ.NTU_TO_J13]
    vis = occ.self_occlusion(xb, L=L)
    check("j13 self_occ: right wrist behind torso hidden", (~vis[:, 6]).all() and vis[:, 7].all())
    z = occ.apply(x, np.ones((64, 13), bool), L=L)
    check("j13 apply: centered on shoulder midpoint", np.allclose(z[0, [10, 11]].mean(0), 0, atol=1e-6))
    for kind in ["none", "random", "lowerbody", "geometric", "combined"]:
        v, f = occ.train_aug(kind, x, rng, L=L)
        check(f"j13 train_aug {kind}: finite", v.shape == (64, 13) and np.isfinite(occ.apply(x, v, f, rng, L=L)).all())
    for name, (fn, fill) in occ.tests(L).items():
        check(f"j13 protocol {name}", fn(x, rng).shape == (64, 13))
    X = np.stack([fake_person(seed=i)[:, occ.NTU_TO_J13] for i in range(16)])
    y = np.arange(16) % 4
    dl = torch.utils.data.DataLoader(TrainSet(X, y, "geometric", L=L), batch_size=8)
    m = MODELS["stgcn"](4, V=13)
    xb_, yb_ = next(iter(dl))
    loss = nn.functional.cross_entropy(m(xb_), yb_)
    loss.backward()
    check("j13 ST-GCN train step finite", torch.isfinite(loss).item())
    acc, f1, mc = evaluate(m, xb_.numpy(), yb_.numpy(), torch.device("cpu"), 4, mca=True)
    check("evaluate mean-class acc in [0,1]", 0 <= mc <= 1)
    # one shoulder hidden (one_arm in the 13-joint layout): center on the visible shoulder
    v = np.ones((64, 13), bool); v[:, list(L.l_arm)] = False
    z = occ.apply(x, v, L=L)
    check("j13 apply: one shoulder hidden -> centered on the other", np.allclose(z[0, 10], 0, atol=1e-6))
    # combined aug hides joints more often than geometric alone
    hg = np.mean([(~occ.train_aug("geometric", x, rng, L=L)[0]).mean() for _ in range(200)])
    hc = np.mean([(~occ.train_aug("combined", x, rng, L=L)[0]).mean() for _ in range(200)])
    check(f"combined aug hides more than geometric ({100 * hc:.0f}% vs {100 * hg:.0f}%)", hc > hg)
    # harmonize: per-frame root at origin (+cam depth), torso = 0.5 m, missing joints stay zero
    Xh = X.copy(); Xh[0, 5, 3] = 0
    H = occ.harmonize(Xh)
    pel = H[:, :, list(L.hips)].mean(2)
    tl = np.linalg.norm(H[:, :, list(L.center)].mean(2) - pel, axis=-1)
    check("harmonize: root (0,0,3.5) every frame, torso 0.5", np.allclose(pel[..., :2], 0, atol=1e-5)
          and np.allclose(pel[..., 2], 3.5, atol=1e-5) and abs(np.median(tl) - 0.5) < 1e-3)
    check("harmonize: missing joints stay zero", np.all(H[0, 5, 3] == 0))


def test_toyota_and_transfer():
    p = ROOT / "data/toyota_all.npz"
    if not p.exists():
        print("SKIP toyota (data/toyota_all.npz missing -- run python -m src.toyota)")
        return
    from src.toyota import load_cs, TO_NTU
    from src.eval_transfer import toyota_transfer_set
    from src.train import load_dataset
    Xtr, ytr, Xte, yte, classes = load_cs()
    check("toyota: 31 classes, CS split sizes", len(classes) == 31 and len(Xtr) > 9000 and len(Xte) > 4000)
    check("toyota: camera depth ~3-4.5 m, y up (head above ankles)",
          2.5 < np.median(Xtr[..., 2]) < 4.5 and (Xtr[:, :, 12, 1] > Xtr[:, :, 0, 1]).mean() > 0.95)
    check("toyota: bone lengths plausible (thigh 0.3-0.55 m)",
          0.3 < np.median(np.linalg.norm(Xtr[:, :, 2] - Xtr[:, :, 4], axis=-1)) < 0.55)
    Xt, yt, targets = toyota_transfer_set()
    check("transfer: 8 shared classes, all present", len(targets) == 8 and set(np.unique(yt)) == set(targets))
    check("transfer: inputs finite, shape (N,3,64,13)", Xt.shape[1:] == (3, 64, 13) and np.isfinite(Xt).all())
    a = load_dataset("ntu13")
    check("ntu13: 13 joints, same clips as ntu", a[0].shape[1:] == (64, 13, 3) and a[4] is occ.J13)
    rng = np.random.default_rng(0)
    for name, (fn, fill) in occ.tests(occ.J13).items():
        hid = [1 - fn(x, rng).mean() for x in Xte[:100]]
        outs = [occ.apply(x, fn(x, rng), fill, rng, L=occ.J13) for x in Xte[:20]]
        check(f"toyota protocol {name:16s} finite (hidden {100 * np.mean(hid):4.1f}%)", np.isfinite(np.stack(outs)).all())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(0)
    test_graph()
    test_models()
    test_occlusion_semantics()
    test_apply()
    test_augs_and_tests()
    test_training_step()
    test_j13()
    if a.real:
        test_real_data()
        test_toyota_and_transfer()
    print(f"\n{sum(OK)}/{len(OK)} checks passed")
    sys.exit(0 if all(OK) else 1)
