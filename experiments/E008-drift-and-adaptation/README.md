---
id: E008
title: "Drift, staleness and exposure re-balancing"
hypotheses: [H9]
status: concluded
conclusion: "Ads rotate fast (13-37 % new creatives a day) and a frozen model loses ~0.003 NE per day, so retrain daily; online recalibration slightly worsens NE on the validation day; the exposure penalty selected on the validation day by a pre-registered rule (lambda = 8) cuts cohort HHI 38 % on test with CTR change +0.27 pp [-0.59, +1.08]."
created-at: 2026-10-01
updated-at: 2026-10-01
---

# E008 — Drift, staleness and exposure re-balancing

**Question:** What drifts across the ten days, what does a stale model cost, and can an adaptation layer spread exposure without losing CTR?

**Method:** Daily PSI against the training window, churn statistics, a staleness backtest of models frozen at different days, a causal per-genre hourly recalibration, and a replay of a cohort-exposure penalty with doubly robust CTR estimates.

**Decision:** Daily retraining with gated promotion; no online recalibration; exposure penalty at lambda = 8 (chosen on validation) pending an online A/B test.

The notebook ([notebook.ipynb](notebook.ipynb), paired with [notebook.py](notebook.py)) holds the code, outputs and reading. Rerun with `uv run poe experiments experiments/E008-drift-and-adaptation/notebook.py`.
