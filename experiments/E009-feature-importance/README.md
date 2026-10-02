---
id: E009
title: "Per-feature importance (LightGBM SHAP and permutation)"
hypotheses: [H2, H7, H8]
status: concluded
conclusion: "The publisher (app, site) and genre carry most of the signal; character ID is used but replaceable by genre (E005); conversation features and character age are within noise."
created-at: 2026-10-02
updated-at: 2026-10-02
---

# E009 — Per-feature importance

**Result:** the publisher and genre carry most of the signal; character ID is used but replaceable; conversation features and character age are within noise.

**Question:** Which individual features does a model rely on, and do the features the ablations dropped look different one by one?

**Method:** One tuned LightGBM on train with every non-text group; on the validation day, mean |SHAP| (how much the model uses a feature) and the log-loss increase when it is shuffled, with an hour-block bootstrap CI (how much the model needs it). Group selection stays with the ablations (E005).

**Decision:** No change to the shipped groups.

The notebook ([notebook.ipynb](notebook.ipynb), paired with [notebook.py](notebook.py)) holds the code, outputs and reading. Rerun with `uv run poe experiments experiments/E009-feature-importance/notebook.py`; the table is [reports/models/importance.csv](../../reports/models/importance.csv).
