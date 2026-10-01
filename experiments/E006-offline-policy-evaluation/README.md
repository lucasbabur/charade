---
id: E006
title: "Offline evaluation of ranking policies"
hypotheses: [H11, H6]
status: concluded
conclusion: "Under reconstructed candidate sets and frequency-share propensities, with DR using an independent LightGBM reward model, the shipped policy's estimated lift over logging is +1.23 pp CTR [+0.20, +2.20] (ungated greedy +1.41 pp; validation day +2.08 pp); ESS ~4 % of rows; a demonstration of the evaluation, not a measured production lift."
created-at: 2026-10-01
updated-at: 2026-10-01
---

# E006 — Offline evaluation of ranking policies

**Question:** Would ranking by the model's pCTR serve more clicked ads than the policy that produced the logs, and what do safety gates and exploration cost?

**Method:** Reconstruct candidate sets per publisher x hour on the test days, take each creative's share of the cell as its logging propensity, re-score every (impression, candidate) pair with the shipped bundle, and estimate each policy with IPS, SNIPS and doubly robust estimators plus paired hour-block bootstrap lifts.

**Decision:** Ship greedy + gates + 5 % exploration from a closed-form distribution, and log every candidate's exact propensity so the next evaluation does not need reconstruction.

The notebook ([notebook.ipynb](notebook.ipynb), paired with [notebook.py](notebook.py)) holds the code, outputs and reading. Rerun with `uv run poe experiments experiments/E006-offline-policy-evaluation/notebook.py`.
