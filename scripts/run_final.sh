#!/usr/bin/env bash
# Final experiments: combined augmentation, 3 seeds (NTU + Toyota), harmonized NTU-13 for transfer.
#   bash scripts/run_final.sh [EPOCHS] [PARALLEL_JOBS]
# Resumable: a run is skipped if its results/<tag>.json already exists (Spot VM preemption).
# Needs data/ntu_dom_xsub.npz, data/toyota_all.npz, data/ntu_dom_natural.npz (licensed data:
# copy only to machines you control).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$(command -v python3 || command -v python)}
EPOCHS=${1:-30}
JOBS=${2:-10}
mkdir -p logs results checkpoints

"$PY" -m tests.smoke_test --real
"$PY" -c "
from src.train import load_dataset, build_test_sets
for ds in ['ntu', 'toyota', 'ntu13', 'ntu13h']:
    _, _, Xva, _, L, cache = load_dataset(ds)
    build_test_sets(Xva, L=L, cache_name=cache)"

job() {  # dataset aug seed
  local ds=$1 aug=$2 seed=$3 tag
  if [ "$ds" = ntu ]; then tag="stgcn_${aug}"; else tag="${ds}_stgcn_${aug}"; fi
  [ "$seed" != 0 ] && tag="${tag}_s${seed}"
  [ -f "results/${tag}.json" ] && return
  echo "$PY -m src.train --dataset $ds --model stgcn --aug $aug --seed $seed --epochs $EPOCHS --workers 2 > logs/${tag}.log 2>&1"
}
{
  for aug in none random lowerbody geometric combined; do job ntu13h $aug 0; done   # transfer first
  for ds in ntu toyota; do job $ds combined 0; done
  for seed in 1 2; do
    for ds in ntu toyota ntu13h; do
      for aug in none random lowerbody geometric combined; do job $ds $aug $seed; done
    done
  done
} | xargs -P "$JOBS" -I{} bash -c "{}"

"$PY" -m src.eval_transfer
"$PY" -m src.eval_natural
"$PY" -m src.report
"$PY" -m src.report_toyota
echo "ALL FINAL RUNS DONE"
