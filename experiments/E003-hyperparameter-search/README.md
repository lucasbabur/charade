---
id: E003
title: "Hyperparameter search for DCN-v2 and LightGBM"
hypotheses: []
status: concluded
conclusion: "Best inner-split NE: DCN-v2 0.8746 (24-dim embeddings, 2 cross layers, 256-128 MLP, dropout 0.3, lr 2.9e-3, batch 2048) vs LightGBM 0.8787; both configs ship in [tool.charade.model]."
created-at: 2026-10-01
updated-at: 2026-10-01
---

# E003 — Hyperparameter search for DCN-v2 and LightGBM

**Question:** Which configurations to carry into the model comparison, chosen without touching the validation day?

**Method:** Optuna TPE, 60 DCN-v2 and 40 LightGBM trials, trained on 10-21..10-26 and selected on 10-27 NE. The validation day (10-28) stays clean for early stopping, calibration and comparisons.

**Decision:** Copy the best trials into `[tool.charade.model]` (done by hand, citing the CSVs).

The notebook ([notebook.ipynb](notebook.ipynb), paired with [notebook.py](notebook.py)) holds the code, outputs and reading. Rerun with `uv run jupytext --execute experiments/E003-hyperparameter-search/notebook.ipynb`.
