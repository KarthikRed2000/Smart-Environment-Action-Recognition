"""Helpers for multi-seed results: <tag>.json is seed 0, <tag>_s<k>.json are further seeds."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (0, 1, 2)
PM = "$\\pm$"


def tags(base):
    return [base if s == 0 else f"{base}_s{s}" for s in SEEDS]


def load(base, folder=ROOT / "results"):
    """List of result dicts for every seed of `base` that exists."""
    out = []
    for t in tags(base):
        f = Path(folder) / f"{t}.json"
        if f.exists():
            out.append(json.loads(f.read_text()))
    return out


def ckpts(base, folder=ROOT / "checkpoints"):
    return [(t, Path(folder) / f"{t}.pt") for t in tags(base) if (Path(folder) / f"{t}.pt").exists()]


def ms(vals):
    """(mean, std) of a list (std = 0 for one value)."""
    v = np.asarray(vals, float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def fmt(vals, scale=100, std=True):
    m, s = ms(vals)
    if std and len(vals) > 1:
        return f"{scale * m:.1f}{PM}{scale * s:.1f}"
    return f"{scale * m:.1f}"
