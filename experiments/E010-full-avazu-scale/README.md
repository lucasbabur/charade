---
id: E010
title: "The same model family at 9x the data (full Kaggle Avazu)"
hypotheses: [H1]
status: concluded
conclusion: "9x the data (every impression of 1 in 4 users of the full Kaggle file) improves NE from 0.914 to 0.857 and AUC from 0.706 to 0.766; DCN-v2 and LightGBM stay within about 0.003 NE of each other at both scales (one seed, no CI), so data volume, not architecture, is the lever."
created-at: 2026-10-02
updated-at: 2026-10-02
---

# E010 — The same model family at 9x the data

**Result:** 9x the data (every impression of 1 in 4 users of the full Kaggle file) improves NE from 0.914 to 0.857 and AUC from 0.706 to 0.766; DCN-v2 and LightGBM stay within about 0.003 NE of each other at both scales (one seed, no CI), so data volume, not architecture, is the lever.

**Question:** Do the NE level and the DCN-v2 vs LightGBM ranking hold when the 1M-row sample is replaced by the full Kaggle Avazu file (same ten days)?

**Method:** Character and conversation groups removed (the full file has no character layer); same counters, temporal split, test window and tuned hyperparameters; one seed. 40M rows need ~150 GB in this in-memory pipeline, so the large run keeps every impression of 1 in 4 users (9.2M rows; user sampling keeps histories exact). Not a leaderboard comparison.

**Decision:** Keep DCN-v2 (ONNX export, character interactions) without claiming it beats boosted trees; re-run the comparison with seeds and CIs at production volume. More data is the first scaling lever.

The notebook ([notebook.ipynb](notebook.ipynb), paired with [notebook.py](notebook.py)) reads [reports/models/fullscale.json](../../reports/models/fullscale.json). Rerun with `uv run python -m charade.analysis.fullscale data/raw/avazu-full/train` after downloading the Kaggle file.
