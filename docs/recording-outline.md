---
title: "Recording outline (~12 minutes)"
created-at: 2026-10-01
updated-at: 2026-10-01
---

# Recording outline (~12 minutes)

Talking points with the artifact to show on screen. Numbers are taken from `reports/`.

| # | Minutes | Point | Show |
|---|---|---|---|
| 1 | 0:00–1:00 | The problem: rank N ads for one chat moment, calibrated, safe, under 50 ms. The approach: hypotheses first, gates always, everything through PRs | README, docs/index.md |
| 2 | 1:00–2:30 | **Data first.** The Avazu sample plus a synthetic character layer. Placeholder devices (82 %). The C-columns hide creative → campaign → advertiser. The same-hour leak and why counters skip the current hour | docs/01-data.md, reports/eda/eda.md |
| 3 | 2:30–3:30 | **Hypotheses before models,** with their prediction and decision rule. Several were refuted or weaker than expected, and I kept them that way | docs/hypotheses.md |
| 4 | 3:30–5:00 | **Model.** Why normalized entropy and calibration (pCTR × bid). DCN-v2 because interactions are real; LightGBM and logistic as yardsticks; paired hour-block CIs | docs/04, reports/models/metrics.json |
| 5 | 5:00–6:00 | **The surprising result:** character ID and description text add nothing. The character *is* its genre and tier, so cold start is solved by construction. A per-character prior would make things worse | ablations.csv, docs/06 |
| 6 | 6:00–7:30 | **Ranking.** Gates first, then value, then Beta uncertainty, greedy plus 5 % Thompson, propensities logged. OPE: +1.28 pp DR for greedy. The shipped policy's CI touches zero; that is the cost of safety and learning, and I say so | docs/05, sample_rankings.json (romance vs new mentor) |
| 7 | 7:30–8:30 | **Drift.** Ads rotate daily; a frozen model loses ~0.003 NE per day, so retrain daily. Online recalibration did nothing (negative result). The exposure penalty: −30 % concentration at no detectable CTR cost | docs/07 |
| 8 | 8:30–10:00 | **Serving.** The request path, exact parity, p99 26 ms at 400 rps. Two bugs the load test found (NaN from out-of-order events; 24 polars threads per worker) | docs/08, latency.json |
| 9 | 10:00–11:00 | **Engineering.** mlcheck (46 gates, recompute rather than trust), CI, Terraform (checkov clean), the daily retrain gate, every commit green | docs/mlcheck.md, GitHub PR list |
| 10 | 11:00–12:00 | **Trade-offs and next steps:** logged propensities, conversation content, creative embeddings, counterfactual training | docs/10 |

**80/20 statement:** most of the time went into data understanding, evaluation discipline and the ranking and serving path. The model architecture itself was a small, tuned DCN, because the data's signal is low-dimensional.
