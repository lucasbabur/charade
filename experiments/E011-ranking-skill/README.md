---
id: E011
title: "Predictive cost of flattening candidate scores"
hypotheses: [H1, H11]
status: concluded
conclusion: "Replacing candidate scores with their unweighted opportunity mean costs 0.0018 logged-impression log loss [0.0010, 0.0030] (NE 0.8842 -> 0.8882). This is a predictive score ablation, not a percentage of ranking skill or an estimate of CTR lift."
created-at: 2026-10-03
updated-at: 2026-10-03
---

# E011 — Predictive cost of flattening candidate scores

**Result:** Replacing candidate scores with their unweighted opportunity mean costs 0.0018 logged-impression log loss [0.0010, 0.0030] (NE 0.8842 -> 0.8882). This is a predictive score ablation, not a percentage of ranking skill or an estimate of CTR lift.

**Question:** How much does logged-impression predictive loss change when every candidate in an opportunity receives the same score?

**Method:** On 102,874 test opportunities, compare the shipped scorer on the reconstructed logged candidate with the unweighted candidate mean. Logged ads outside the top-K are excluded; ad attributes are reconstructed by their mode. Paired hour-block bootstrap of the logged-impression log-loss gap.

**Limits:** Averaging changes calibration and candidate weighting as well as discrimination. With nonuniform logging, an unweighted candidate mean is not the context-conditional CTR. Duplicating a lower-scored alternative changes this gap without changing the chosen ad. The earlier skill-share interpretation is withdrawn; policy value still needs the assumptions in [04](../../docs/04-ranking-policy.md) or an online test.

**Decision:** No model change. Report the score-ablation gap with its scope; evaluate ranking decisions with real candidate sets, propensities and outcomes.

The notebook ([notebook.ipynb](notebook.ipynb), paired with [notebook.py](notebook.py)) runs `charade.analysis.opportunity` and writes [reports/models/opportunity.json](../../reports/models/opportunity.json).
