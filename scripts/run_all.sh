#!/usr/bin/env bash
# Full experiment matrix. Run from the repo root:   bash scripts/run_all.sh [EPOCHS] [PARALLEL_JOBS]
# ~3M-param models are small, so several jobs can share one GPU (PARALLEL_JOBS=4 is fine on a T4/L4).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$(command -v python3 || command -v python)}
EPOCHS=${1:-30}
JOBS=${2:-4}
mkdir -p logs results checkpoints

# 1) data (skip steps that are already done)
if [ ! -f data/raw/ntu60_3danno.pkl ]; then
  mkdir -p data/raw
  curl -L --retry 3 -o data/raw/ntu60_3danno.pkl \
    https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu60_3danno.pkl
fi
[ -f data/ntu_dom_xsub.npz ] || "$PY" -m src.data

# 2) smoke test, then build the fixed corrupted test sets once (avoids parallel jobs racing on the cache)
"$PY" -m tests.smoke_test --real
"$PY" -c "import numpy as np; from src.train import build_test_sets; build_test_sets(np.load('data/ntu_dom_xsub.npz')['X_val'])"

# 3) experiments: ablation on ST-GCN + baselines with/without our augmentation
RUNS=(
  "stgcn none" "stgcn random" "stgcn lowerbody" "stgcn geometric"
  "bilstm none" "bilstm geometric"
  "cnn1d none" "cnn1d geometric"
)
for r in "${RUNS[@]}"; do
  set -- $r
  echo "$PY -m src.train --model $1 --aug $2 --epochs $EPOCHS --workers 3 > logs/$1_$2.log 2>&1"
done | xargs -P "$JOBS" -I{} bash -c "{}"

# 4) tables
"$PY" -m src.report
