# Occlusion-robust ST-GCN (CS771)

Skeleton action recognition on a 22-class domestic subset of NTU RGB+D 60 (cross-subject),
with **geometry-based occlusion** (virtual furniture in 3D + self-occlusion behind the torso,
zero or sensor-like noisy fill) compared against no / random-joint / lower-body masking.

```
src/data.py        safe-load NTU-60 3D pickle (OpenMMLab), build subset -> data/ntu_dom_xsub.npz
src/occlusion.py   occlusion models, training augmentations, 8 test protocols
src/models.py      ST-GCN, Bi-LSTM, 1D-CNN
src/train.py       train one config, evaluate on all protocols -> results/<model>_<aug>.json
src/report.py      results -> Markdown + results/table.tex
tests/smoke_test.py  69 fast checks (~20 s), no full training
scripts/run_all.sh   data + smoke test + 8 training runs + tables
scripts/gcp_setup.sh create GPU VM on GCP and launch run_all.sh there
```

Quick start (any machine with a GPU):

```bash
pip install -r requirements.txt
python -m tests.smoke_test            # add --real once data/ntu_dom_xsub.npz exists
bash scripts/run_all.sh 30 4          # epochs, parallel jobs
```

Test protocols: `clean`, `lowerbody`, `furniture`, `furniture_noisy`, `self_occ` and, never used
in training, `one_arm`, `temporal_block`, `random_30`. If you change `occlusion.py`, delete
`data/test_sets.npz` so the fixed test sets are rebuilt.
