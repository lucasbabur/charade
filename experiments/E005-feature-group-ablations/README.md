---
id: E005
title: "Feature-group ablations"
hypotheses: [H2, H3, H4, H7]
status: concluded
conclusion: "Character metadata is worth +0.0035 log loss and user history +0.0012; character ID and conversation features are within noise and were dropped; text features hurt (+0.0005 to +0.0007)."
created-at: 2026-10-01
updated-at: 2026-10-03
---

# E005 — Feature-group ablations

**Question:** Which feature groups earn their place in the model?

**Method:** 3-seed DCN-v2 ensembles on the validation day, each identical to the all-features reference except for one change; paired hour-block bootstrap of the log-loss difference (raw logits, because the calibrator is fitted on validation).

**Decision:** Ship without character ID and conversation groups; reject text; keep metadata, device and user history.

The notebook ([notebook.ipynb](notebook.ipynb), paired with [notebook.py](notebook.py)) holds the code, outputs and reading. Rerun with `uv run poe experiments experiments/E005-feature-group-ablations/notebook.py`.
