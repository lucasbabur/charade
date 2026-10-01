---
id: E006
title: "Offline evaluation of ranking policies"
hypotheses: [H11, H6]
status: concluded
conclusion: "Greedy pCTR ranking beats the logging policy by +1.28 pp CTR (DR, [+0.07, +2.50]); the shipped policy with gates and 5 % exploration is +1.13 pp with a CI touching zero; ESS ~4 % of rows."
created-at: 2026-10-01
updated-at: 2026-10-01
---

# E006 — Offline evaluation of ranking policies

**Question:** Would ranking by the model's pCTR serve more clicked ads than the policy that produced the logs, and what do safety gates and exploration cost?

**Method:** Reconstruct candidate sets per publisher x hour on the test days, take each creative's share of the cell as its logging propensity, re-score every (impression, candidate) pair with the shipped bundle, and estimate each policy with IPS, SNIPS and doubly robust estimators plus paired hour-block bootstrap lifts.

**Decision:** Ship greedy + gates + 5 % Thompson exploration and log real propensities and candidate sets so the next evaluation does not need reconstruction.

The notebook ([notebook.ipynb](notebook.ipynb), paired with [notebook.py](notebook.py)) holds the code, outputs and reading. Rerun with `uv run poe experiments experiments/E006-offline-policy-evaluation/notebook.py`.
