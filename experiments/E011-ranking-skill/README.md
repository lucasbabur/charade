---
id: E011
title: "How much of the model's skill ranks ads?"
hypotheses: [H1, H11]
status: concluded
conclusion: "Only 3.4 % of the model's skill ranks ads: replacing each candidate's pCTR with its opportunity's mean costs 0.0018 log loss [0.0010, 0.0030] (NE 0.8842 -> 0.8882); the rest predicts whether a moment is clickable at all, which the ranker cannot use."
created-at: 2026-10-03
updated-at: 2026-10-03
---

# E011 — How much of the model's skill ranks ads?

**Result:** Only 3.4 % of the model's skill ranks ads: replacing each candidate's pCTR with its opportunity's mean costs 0.0018 log loss [0.0010, 0.0030] (NE 0.8842 -> 0.8882); the rest predicts whether a moment is clickable at all, which the ranker cannot use.

**Question:** NE rewards predicting whether a moment gets a click; the ranker only uses differences between candidates of one moment. How big is that second part?

**Method:** On 102,874 test opportunities (impressions with their reconstructed candidate sets, as in off-policy evaluation), compare the logged ad's shipped pCTR with the mean pCTR of its opportunity's candidates; paired hour-block bootstrap. Suggested by the external reviewer's flattening diagnostic.

**Decision:** No model change; report the within-opportunity gap next to NE, and prioritise ad-content and conversation features, which act within opportunities.

The notebook ([notebook.ipynb](notebook.ipynb), paired with [notebook.py](notebook.py)) runs `charade.analysis.opportunity` and writes [reports/models/opportunity.json](../../reports/models/opportunity.json).
