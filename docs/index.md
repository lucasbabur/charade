# Documentation index

Status: ✅ written · 🟡 partial · ⬜ planned. Update the status in the same commit that changes a doc.

## Start here

| Doc | Status | What it answers |
|---|---|---|
| [../README.md](../README.md) | 🟡 | What this is; how to install, reproduce and serve |
| [../AGENTS.md](../AGENTS.md) | ✅ | Rules, invariants, commands and the definition of done for AI agents (and humans) |
| [PLAN.md](PLAN.md) | ✅ | Full design: data findings, split, features, models, evaluation, ranking, cold start, drift, serving, infrastructure, CI, commit sequence |
| [mlcheck.md](mlcheck.md) | ✅ | The 46 ML release gates, the artifact contract, the statistics, first results on the data |
| 00-summary.md | ⬜ | One page: problem, decisions, headline results with CIs, next steps |

## By topic

| Doc | Status | PLAN § | What it answers |
|---|---|---|---|
| 01-data.md | ⬜ | 0, 3 | What the data is (Avazu + synthetic characters), contract, placeholders, C-feature hierarchy |
| 02-features.md | ⬜ | 4, 5 | Every feature: definition, train/serve source, leakage notes, ablation Δ |
| 03-text-enrichment.md | ⬜ | 6 | Template finding, embedding bake-off, Claude attributes, safety cross-check |
| 04-models-evaluation.md | ⬜ | 7, 8 | Split rationale, models, metrics with CIs, slices, calibration, ablations, OPE |
| 05-ranking-policy.md | ⬜ | 9 | Gates, EV, fatigue, pacing, exploration and propensities, ordering under uncertainty |
| 06-cold-start.md | ⬜ | 10 | Signals before the first click, priors, graduation rule |
| 07-drift-adaptation.md | ⬜ | 11 | Temporal shifts, staleness cost, adaptation simulator results |
| 08-serving-architecture.md | ⬜ | 12, 13 | Request path, latency budget against measured numbers, cache/precompute/approximate, observability |
| 09-operations.md | ⬜ | 13, 14, 17 | Monitors, alarms, runbooks, retraining and promotion, CI/CD |
| 10-next-steps.md | ⬜ | 20 | Data to gather, models to try, scaling |
| adr/ | ⬜ | all | One decision per file: context, decision, consequences |
| recording-outline.md | ⬜ | 20 | Script for the video walkthrough |

## Reading paths

| Task | Read, in order |
|---|---|
| Change a feature | AGENTS.md invariants 2–4 → 02-features → PLAN §4–5 → `src/cameo/features/` |
| Train or compare models | PLAN §1, §7–8 → 04-models-evaluation → mlcheck.md (model stage) → `configs/experiments/` |
| Touch the ranking policy | 05-ranking-policy → PLAN §9 → mlcheck MLP* |
| Touch the API or latency | 08-serving-architecture → PLAN §12–13 → AGENTS.md invariant 5 |
| Add an ML gate | tools/mlcheck/AGENTS.md → mlcheck.md |
| Infrastructure | PLAN §14, §17 → 09-operations → `infra/terraform/` |
| Review the project in 10 minutes | 00-summary → mlcheck.md results → 04-models-evaluation → 05-ranking-policy |

## Generated, not written

`reports/` (figures, tables, `sample_rankings.json`) is produced by CLI commands and committed. Docs cite it and never retype numbers.
