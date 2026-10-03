---
id: E010
title: "The same model family at 9x the data (full Kaggle Avazu)"
hypotheses: [H1]
status: concluded
conclusion: "On one shared test set, 9x the training data (0.74M -> 6.7M rows from the full Kaggle file) lowers NE by 0.016 (DCN-v2 0.873 -> 0.857, LightGBM 0.875 -> 0.855); the two models swap places and stay within 0.003 NE, so volume moves NE about four times more than the architecture choice."
created-at: 2026-10-02
updated-at: 2026-10-03
---

# E010 — The same model family at 9x the data

**Result:** On one shared test set, 9x the training data (0.74M -> 6.7M rows from the full Kaggle file) lowers NE by 0.016 (DCN-v2 0.873 -> 0.857, LightGBM 0.875 -> 0.855); the two models swap places and stay within 0.003 NE, so volume moves NE about four times more than the architecture choice.

**Question:** Do the NE level and the DCN-v2 vs LightGBM ranking hold when the 1M-row sample is replaced by the full Kaggle Avazu file (same ten days)?

**Method:** Character and conversation groups removed (the full file has no character layer). From the full file, 1 in 4 users (exact histories); models trained on 1 in 36 users (0.74M rows) and on all of them (6.7M rows), **both scored on the same 1.2M test rows**; one seed. The sample is also trained without characters and scored on its own test set. Not a leaderboard comparison. Run after every decision was made; nothing shipped was trained, tuned or selected on this file (it covers the test days, so doing so would leak). An earlier version compared different test sets and overstated the volume effect (0.057).

**Decision:** Keep DCN-v2 (ONNX export, character interactions) without claiming it beats boosted trees; re-run the comparison with seeds and CIs at production volume. More data is the first scaling lever.

The notebook ([notebook.ipynb](notebook.ipynb), paired with [notebook.py](notebook.py)) reads [reports/models/fullscale.json](../../reports/models/fullscale.json). Rerun with `uv run python -m charade.analysis.fullscale data/raw/avazu-full/train` after downloading the Kaggle file.
