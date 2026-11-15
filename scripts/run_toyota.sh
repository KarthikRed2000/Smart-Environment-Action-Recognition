#!/usr/bin/env bash
# Toyota Smarthome benchmark + NTU->Toyota transfer.  Run from the repo root:
#   bash scripts/run_toyota.sh [EPOCHS] [PARALLEL_JOBS]
# Needs data/ntu_dom_xsub.npz and data/toyota_all.npz (built by src.data / src.toyota;
# both datasets are licensed -- copy them only to machines you control).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$(command -v python3 || command -v python)}
EPOCHS=${1:-30}
JOBS=${2:-8}
mkdir -p logs results checkpoints

"$PY" -m tests.smoke_test --real
# build the fixed corrupted test sets once per dataset (avoids parallel jobs racing on the cache)
"$PY" -c "
from src.train import load_dataset, build_test_sets
for ds in ['toyota', 'ntu13']:
    _, _, Xva, _, L, cache = load_dataset(ds)
    build_test_sets(Xva, L=L, cache_name=cache)"

for ds in toyota ntu13; do
  for aug in none random lowerbody geometric; do
    echo "$PY -m src.train --dataset $ds --model stgcn --aug $aug --epochs $EPOCHS --workers 2 > logs/${ds}_stgcn_${aug}.log 2>&1"
  done
done | xargs -P "$JOBS" -I{} bash -c "{}"

"$PY" -m src.eval_transfer
"$PY" -m src.report_toyota
