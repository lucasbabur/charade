---
id: E008
title: "Drift, staleness and exposure re-balancing"
hypotheses: [H9]
status: concluded
conclusion: "Ads rotate fast (13-37 % new creatives a day) and a frozen model loses ~0.003 NE per day, so retrain daily; online recalibration adds nothing; an exposure penalty (lambda = 4) cuts cohort concentration 30 % with no detectable CTR loss."
created-at: 2026-10-01
updated-at: 2026-10-01
---

# E008 — Drift, staleness and exposure re-balancing

**Question:** What drifts across the ten days, what does a stale model cost, and can an adaptation layer spread exposure without losing CTR?

**Method:** Daily PSI against the training window, churn statistics, a staleness backtest of models frozen at different days, a causal per-genre hourly recalibration, and a replay of a cohort-exposure penalty with doubly robust CTR estimates.

**Decision:** Daily retraining with gated promotion; no online recalibration; exposure penalty recommended at lambda = 4 pending an online A/B test.

The notebook ([notebook.ipynb](notebook.ipynb), paired with [notebook.py](notebook.py)) holds the code, outputs and reading. Rerun with `uv run jupytext --execute experiments/E008-drift-and-adaptation/notebook.ipynb`.
