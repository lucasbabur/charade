---
title: "Documentation index"
created-at: 2026-10-01
updated-at: 2026-10-01
---

# Documentation index

Start with [00-summary.md](00-summary.md). Freshness is in each doc's frontmatter (`updated-at`).

## Glossary

| Term | Meaning |
|---|---|
| pCTR | Predicted click probability, calibrated so that pCTR × bid is a fair value |
| NE | Normalized entropy: log loss ÷ entropy of the set's base rate. 1.0 = predicting the base rate; lower is better |
| ECE | Expected calibration error over equal-mass bins |
| Hour-block bootstrap | Confidence interval that resamples whole hours, because rows in an hour are correlated |
| Propensity | Probability that the serving policy showed the ad it showed; needed to evaluate other policies on logged data |
| OPE | Off-policy evaluation: estimating a new policy's CTR from logs of another policy |
| IPS / SNIPS / DR | OPE estimators: inverse-propensity weighting, its self-normalized form, and doubly robust (model estimate + weighted correction) |
| ESS | Effective sample size of an importance-weighted estimate |
| PSI | Population stability index; > 0.25 means a feature's distribution moved materially |

## Docs

| Doc | Answers |
|---|---|
| [00-summary.md](00-summary.md) | Problem, headline results, what the data taught, limits |
| [hypotheses.md](hypotheses.md) | Pre-registered hypotheses, metrics, decision rules and outcomes |
| [01-data.md](01-data.md) | Data contract, findings, split, fixture |
| [02-features.md](02-features.md) | Shipped and rejected features, encoding rules |
| [03-text-enrichment.md](03-text-enrichment.md) | Description embedding bake-off |
| [04-models-evaluation.md](04-models-evaluation.md) | Protocol, models, ablations, slices, leakage probes |
| [05-ranking-policy.md](05-ranking-policy.md) | Gates, value, exploration, propensities, offline policy evaluation, sample rankings |
| [06-cold-start.md](06-cold-start.md) | New characters and users, pre-click signal, graduation |
| [07-drift-adaptation.md](07-drift-adaptation.md) | Temporal shifts, staleness, recalibration, exposure re-balancing |
| [08-serving-architecture.md](08-serving-architecture.md) | Request path, latency budget, failure modes |
| [09-operations.md](09-operations.md) | AWS topology, delivery, alarms, runbooks |
| [10-next-steps.md](10-next-steps.md) | Data, models, scaling |
| [mlcheck.md](mlcheck.md) | The 46 ML release gates and their artifact contract |
| [adr/](adr/) | One decision per file |
| [api/openapi.json](api/openapi.json) | Generated API contract (CI fails if stale) |
| [recording-outline.md](recording-outline.md) | Video walkthrough script |

## Experiments

One folder per experiment in [experiments/](../experiments/): `README.md` (frontmatter: id, title, hypotheses, status, conclusion, dates) and a notebook paired with a `.py` script. CI executes every notebook on the fixture (`uv run poe test-notebooks`); `uv run poe experiments` reruns them on the full data.

| Id | Experiment |
|---|---|
| [E001](../experiments/E001-eda-signal-survey/README.md) | Signal survey on the training split |
| [E002](../experiments/E002-text-embedding-bakeoff/README.md) | Character description embeddings |
| [E003](../experiments/E003-hyperparameter-search/README.md) | Hyperparameter search for DCN-v2 and LightGBM |
| [E004](../experiments/E004-model-comparison/README.md) | CTR model comparison on the test days |
| [E005](../experiments/E005-feature-group-ablations/README.md) | Feature-group ablations |
| [E006](../experiments/E006-offline-policy-evaluation/README.md) | Offline evaluation of ranking policies |
| [E007](../experiments/E007-cold-start-and-graduation/README.md) | Cold start and graduation |
| [E008](../experiments/E008-drift-and-adaptation/README.md) | Drift, staleness and exposure re-balancing |

## Generated reports

| Report | Command |
|---|---|
| [reports/eda/eda.md](../reports/eda/eda.md) | `uv run poe eda` |
| [reports/text_bakeoff.csv](../reports/text_bakeoff.csv) | `uv run poe text` |
| [reports/tuning/](../reports/tuning/) | `uv run poe tune` |
| [reports/models/metrics.json](../reports/models/metrics.json), [ablations.csv](../reports/models/ablations.csv) | `uv run poe train`, `uv run poe ablate` |
| [reports/policy/ope.md](../reports/policy/ope.md), [sample_rankings.json](../reports/sample_rankings.json) | `uv run poe ope`, `uv run poe samples` |
| [reports/coldstart/coldstart.md](../reports/coldstart/coldstart.md) | `uv run poe coldstart` |
| [reports/drift/drift.md](../reports/drift/drift.md), [adaptation.md](../reports/drift/adaptation.md) | `uv run poe drift`, `uv run poe adapt` |
| [reports/serving/latency.json](../reports/serving/latency.json) | `uv run poe loadtest` |

## Reading paths

| Task | Read, in order |
|---|---|
| Review the project in 10 minutes | 00-summary → hypotheses → 04 → 05 → adr/ |
| Change a feature | AGENTS.md invariants 2–4 → 02 → `src/charade/features/` |
| Train or compare models | 04 → mlcheck.md (model stage) → `[tool.charade.model]` |
| Touch the ranking policy | 05 → mlcheck.md (policy stage) |
| Touch the API or latency | 08 → AGENTS.md invariant 5 |
| Add an ML gate | tools/mlcheck/AGENTS.md → mlcheck.md |
| Infrastructure | 09 → `infra/terraform/` |
