---
id: E004
title: "CTR model comparison on the test days"
hypotheses: [H1, H9]
status: concluded
conclusion: "DCN-v2 (3-seed ensemble, isotonic) reaches test NE 0.8848 and beats LightGBM by -0.0040 [-0.0052, -0.0027] and logistic by -0.0120 [-0.0138, -0.0102] log loss; calibration holds on both test days."
created-at: 2026-10-01
updated-at: 2026-10-01
---

# E004 — CTR model comparison on the test days

**Question:** Does a cross network beat additive and tree baselines on the same inputs, and does validation-day calibration transfer to the test days?

**Method:** Train logistic, tuned LightGBM and a 3-seed DCN-v2 ensemble on 10-21..10-27 with early stopping on 10-28; choose identity/Platt/isotonic calibration by hour-fold CV on 10-28; score the test days once; compare with a paired hour-block bootstrap. Writes the serving bundle and the mlcheck evidence.

**Decision:** Ship the DCN-v2 ensemble; keep LightGBM and logistic as yardsticks in every run.

The notebook ([notebook.ipynb](notebook.ipynb), paired with [notebook.py](notebook.py)) holds the code, outputs and reading. Rerun with `uv run poe experiments experiments/E004-model-comparison/notebook.py`.
