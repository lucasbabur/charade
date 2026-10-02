---
id: E004
title: "CTR model comparison on the test days"
hypotheses: [H1, H9]
status: concluded
conclusion: "DCN-v2 3-seed ensemble reaches test NE 0.8855 (pred/obs 1.002, no calibration map needed) and beats an equally tuned 3-seed LightGBM ensemble by -0.0040 [-0.0053, -0.0027] and logistic by -0.0117; single model vs single model DCN-LightGBM is -0.0020 [-0.0041, +0.0004], not established."
created-at: 2026-10-01
updated-at: 2026-10-02
---

# E004 — CTR model comparison on the test days

**Question:** Does a cross network beat additive and tree baselines on the same inputs, and does validation-day calibration transfer to the test days?

**Method:** Train logistic, tuned LightGBM and a 3-seed DCN-v2 ensemble on 10-21..10-27 with early stopping on 10-28; choose identity/Platt/isotonic calibration by hour-fold CV on 10-28; score the test days once; compare with a paired hour-block bootstrap. Writes the serving bundle and the mlcheck evidence.

**Decision:** Ship the DCN-v2 ensemble (ensemble result + single ONNX graph); keep LightGBM and logistic as yardsticks with equal tuning and seeds; do not claim an architecture advantage over trees.

The notebook ([notebook.ipynb](notebook.ipynb), paired with [notebook.py](notebook.py)) holds the code, outputs and reading. Rerun with `uv run poe experiments experiments/E004-model-comparison/notebook.py`.
