---
id: E012
title: "Live (campaign, genre) correction: graduation and adaptation from outcomes"
hypotheses: [H12]
status: concluded
conclusion: "Selected on validation (prior 80 expected clicks, no decay), the corrected greedy policy is non-inferior to greedy pCTR on test at the 0.2 pp margin with a positive point estimate, lowers the log loss of the shown ad with a CI excluding zero, and narrows the cold-campaign under-prediction. Graduation is defined: a (campaign, genre) pair has graduated once its logged expected clicks reach the prior. Half-life is indistinguishable on 30 hours."
created-at: 2026-10-03
updated-at: 2026-10-03
---

# E012 — Live (campaign, genre) correction: graduation and adaptation from outcomes

**Question:** Can one online component answer the two open halves of the brief, a rule for when a cold entity has graduated and an adaptation layer that follows outcomes, without hurting CTR?

**Method:** A Gamma-Poisson posterior per (campaign, genre) on a multiplier of the model's pCTR, learned from logged outcomes in strictly earlier hours and replayed over the reconstructed candidate sets of E006. Prior strength, half-life and an exploration bonus form the grid. Selection on the validation day by the E008 non-inferiority rule, one read of test. Two readings: the doubly robust CTR change against greedy pCTR, and the log loss of the corrected pCTR on the ad actually shown, which needs no off-policy assumption.

**Result (test, read once):** DR CTR change vs greedy <!--n:corr_dr-->+0.49 pp [−0.07, +1.02]<!--/n-->; log loss of the corrected vs raw pCTR on the shown ad <!--n:corr_logloss-->−0.00054 [−0.00087, −0.00022]<!--/n--> (NE <!--n:corr_ne-->0.8842 → 0.8831<!--/n-->); cold-campaign pred/obs <!--n:corr_cold_pred_obs-->0.937 → 0.951<!--/n-->; <!--n:corr_graduated-->23 %<!--/n--> of impressions in the last test hour fell on graduated pairs from an empty start; cohort HHI <!--n:corr_hhi-->0.0275<!--/n--> against 0.0293 for greedy ([adaptation.md](../../reports/drift/adaptation.md)). Full grid in [reports/policy/correction.md](../../reports/policy/correction.md).

**Decision:** Ship the correction in the policy with prior <!--n:corr_prior-->80<!--/n--> and no decay, replacing the frozen training-count evidence table; graduation = logged expected clicks ≥ prior. The CTR point estimate awaits the same online test as the exposure penalty.

The notebook ([notebook.ipynb](notebook.ipynb), paired with [notebook.py](notebook.py)) holds the code, outputs and reading. Rerun with `uv run poe experiments experiments/E012-live-correction/notebook.py`.
