---
id: E001
title: "Signal survey on the training split"
hypotheses: [H2, H5, H6, H7, H8]
status: concluded
conclusion: "Genre, safety tier, prior user activity and campaign repetition carry signal; conversation turn and character age do not; same-hour user counts are a leak."
created-at: 2026-10-01
updated-at: 2026-10-01
---

# E001 — Signal survey on the training split

**Question:** Which raw signals move CTR before any model, and which of them would leak?

**Method:** Group-by CTR tables on the training split only (validation and test unseen), plus the genre/tier x campaign interaction residuals against their binomial noise.

**Decision:** Keep genre, tier and causal user counters; treat turn and age as ablation candidates; exclude same-hour counts.

The notebook ([notebook.ipynb](notebook.ipynb), paired with [notebook.py](notebook.py)) holds the code, outputs and reading. Rerun with `uv run jupytext --execute experiments/E001-eda-signal-survey/notebook.ipynb`.
