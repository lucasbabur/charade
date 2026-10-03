---
title: "Recording outline (~12 minutes)"
created-at: 2026-10-01
updated-at: 2026-10-03
---

# Recording outline (~12 minutes)

Talking points with the artifact to show on screen. Numbers are taken from `reports/`.

| # | Minutes | Point | Show |
|---|---|---|---|
| 1 | 0:00–1:00 | The problem: rank N ads for one chat moment, calibrated, safe, under 50 ms. The approach: hypotheses first, gates always, everything through PRs | README, docs/index.md |
| 2 | 1:00–2:30 | **Data first.** The Avazu sample plus a synthetic character layer. Placeholder devices (82 %). The C-columns hide creative → campaign → advertiser. The same-hour leak and why counters skip the current hour | docs/01-data.md, reports/eda/eda.md |
| 3 | 2:30–3:30 | **Hypotheses before models,** with their prediction and decision rule. Several were refuted or weaker than expected, and I kept them that way | docs/hypotheses.md |
| 4 | 3:30–5:00 | **Model.** NE and calibration (pCTR × bid), model comparisons with paired hour-block CIs. E010 compares two data volumes on shared test rows but has one seed and no comparison intervals. E011 measures the predictive loss cost of flattening scores; it does not decompose ranking skill or establish policy lift | docs/03, reports/models/metrics.json |
| 5 | 5:00–6:00 | **The surprising result:** on this data character ID and description text add nothing beyond genre and tier, so the shipped model has no character ID and a new character has no missing parameter. Caveat: synthetic layer, templated text; the real problem is semantic matching, which needs ad content. Cold ads (43 % of test) are the bigger cold start | ablations.csv, importance.csv, docs/05 |
| 6 | 6:00–7:30 | **Ranking.** Gates first, then value, then greedy plus 5 % exploration from a known distribution, exact propensities for every candidate. The bias the external review found in the first (Thompson) version and the recovery test that now guards it. OPE: +1.46 pp DR, under stated assumptions; exploration costs 0.06 pp | docs/04, sample_rankings.json |
| 7 | 7:30–8:30 | **Drift and adaptation.** The live (campaign, genre) correction (E012): selected on validation, non-inferior on test, better log loss on the shown ad, and a graduation rule for cold ads. Three-day rolling windows still vary in row count and traffic mix. Report NE, AUC and calibration descriptively; neither a causal ageing rate nor ranking stability is established. Daily retraining is an operational default. The validation-selected exposure penalty lowers concentration, but test does not confirm CTR non-inferiority | docs/06 |
| 8 | 8:30–10:00 | **Serving.** The request path, exact parity, p99 25–31 ms at 400 rps. Two bugs the load test found (NaN from out-of-order events; 24 polars threads per worker) | docs/07, latency.json |
| 9 | 10:00–11:00 | **Engineering.** mlcheck (44 gates, recompute rather than trust), CI and a tag-triggered image release, a Terraform sketch (validated, never applied), gated retrain with pinned-bundle rollback, and what was cut as overbuilt | docs/mlcheck.md, GitHub PR list |
| 10 | 11:00–12:00 | **Next steps, in order:** real logged candidate sets and propensities, ad content features for the 43 % unseen creatives, conversation context | docs/08 |

**80/20 statement:** most of the time went into data understanding, evaluation discipline and the ranking and serving path. The model architecture itself was a small, tuned DCN, because the data's signal is low-dimensional.
