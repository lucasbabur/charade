---
id: E008
title: "Drift, staleness and exposure re-balancing"
hypotheses: [H9]
status: concluded
conclusion: "The rolling-window backtest reports NE, AUC and calibration with training counts; varying volume and traffic mix prevent isolating model age, and no ranking-stability claim is established. Online recalibration slightly worsens validation NE. The selected exposure penalty (lambda = 2) cuts cohort HHI 24 % on test at +0.06 pp CTR [-0.60, +0.74], which does not confirm non-inferiority at the 0.2 pp margin."
created-at: 2026-10-01
updated-at: 2026-10-03
---

# E008 — Drift, staleness and exposure re-balancing

**Question:** What drifts across the ten days, what does a stale model cost, and can an adaptation layer spread exposure without losing CTR?

**Method:** Daily PSI against the training window, churn statistics, a staleness backtest of models frozen at different days, a causal per-genre hourly recalibration, and a replay of a cohort-exposure penalty with doubly robust CTR estimates.

**Decision:** Daily retraining with gated promotion; no online recalibration; exposure penalty at lambda = 2 (chosen on validation) is a candidate for an online A/B test, not shipped: test does not confirm non-inferiority.

The notebook ([notebook.ipynb](notebook.ipynb), paired with [notebook.py](notebook.py)) holds the code, outputs and reading. Rerun with `uv run poe experiments experiments/E008-drift-and-adaptation/notebook.py`.
