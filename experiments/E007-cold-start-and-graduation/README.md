---
id: E007
title: "Cold start and graduation"
hypotheses: [H10, H8]
status: concluded
conclusion: "Characters carry no measurable CTR spread beyond genre x tier, so no character ever graduates; unseen characters score NE 0.906 vs 0.885 for warm ones (n = 1,478, no CI)."
created-at: 2026-10-01
updated-at: 2026-10-01
---

# E007 — Cold start and graduation

**Question:** What signal exists before an entity's first click, and when should a character get its own parameters?

**Method:** Permutation importance on cold slices of the test days; method-of-moments Beta priors per genre x tier with binomial noise removed; a causal per-character logit correction from strictly earlier impressions, evaluated by earlier-impression bucket.

**Decision:** No character ID in the model; graduation is re-measured on fresh data instead of fixed as a constant.

The notebook ([notebook.ipynb](notebook.ipynb), paired with [notebook.py](notebook.py)) holds the code, outputs and reading. Rerun with `uv run poe experiments experiments/E007-cold-start-and-graduation/notebook.py`.
