---
title: "mlcheck — ML release gates"
created-at: 2026-10-01
updated-at: 2026-10-01
---

# mlcheck — ML release gates

`tools/mlcheck` is a workspace package that answers one question before a model ships: **is this run leak-free, reproducible, better than the baseline with a CI, calibrated, identical between train and serve, fast enough, and safe to serve?** Exit code 1 on any blocking failure; CI runs it as a required job.

## Why custom

| Option | Verdict |
|---|---|
| deepchecks | Closest scope (data integrity, train/test validation), but last release 2024-12; no parity, latency, OPE or holdout discipline |
| Evidently, Great Expectations, pandera | Maintained; cover drift and data quality only. pandera stays as the in-pipeline schema; Evidently may compute the PSI that `drift.json` carries |
| import-linter | Correct tool for import rules; mlcheck uses its engine (**grimp**) directly to report the offending import chain |

mlcheck is ~900 lines of glue and gates. It **recomputes model metrics from raw predictions** rather than trusting the project's reports, so a bug in the evaluation code cannot pass its own gate.

## Usage

```bash
uv run mlcheck .                                  # every stage
uv run mlcheck . --stage data --stage static      # what works before a run exists
uv run mlcheck . --only MLM004 -v                 # one gate with evidence
uv run mlcheck . --format json --strict           # CI: warnings block too
uv run mlcheck --list                             # catalogue
```

Configuration: `[tool.mlcheck]` in `pyproject.toml` (see the repo root). Every threshold lives in `[tool.mlcheck.thresholds]`.

Semantics: missing artifact or missing source package → **FAIL** (never a vacuous pass). Unconfigured section → SKIP. A crashing check → FAIL with the exception. WARNING-severity checks report WARN and block only with `--strict`. Static findings can be suppressed per line with `# mlcheck: ignore[CODE]`, which leaves a greppable, reviewable trail (e.g. the calibrator fitted on validation data).

## Artifact contract (`artifacts/current/`)

Pydantic models in `mlcheck.contract`; the training pipeline writes them with `Model(...).model_dump_json()`.

| File | Content | Gates |
|---|---|---|
| `manifest.json` | run id, git sha + dirty flag, config hash, sha256 per data file, split windows, every fitted artifact with `fit_split`/`fit_end`, seeds per model, library versions | MLR*, MLL002–003 |
| `splits.parquet` | `id, split, ts` | MLL001–002 |
| `predictions.parquet` | `id, split, model, label, pred, ts, slice_<name>…` for primary **and** baseline on val + test | MLM* |
| `leakage.json` | shuffled-label AUC, per-feature univariate AUC, adversarial AUC | MLL004–005, MLL007 |
| `evaluation_ledger.jsonl` | one line per evaluation: run, model version, split | MLL006 |
| `parity.json` | offline vs serving features, torch vs ONNX outputs | MLV001–002 |
| `latency.json` | load-test p50/p95/p99, error rate, N candidates | MLV003–004 |
| `ope.json` | per policy and estimator: value, CI, ESS, max weight | MLP001–002 |
| `decisions.jsonl` | sampled decision logs: candidates, gates, chosen id, propensity, exploration flag | MLP003–004 |
| `drift.json` | PSI per feature per period vs reference | MLX001 |

## Statistics used

- **Normalized entropy** = log loss ÷ entropy of the evaluation set's base rate. Below 1 beats the constant predictor; this is the metric the auction consumes.
- **Beats baseline** = paired per-row log-loss difference, percentile CI from an **hour-block bootstrap** (1,000 resamples). Rows within an hour share traffic mix, so a row-level bootstrap would give intervals that are too narrow. A unit test shows the block CI is >5× the naive one under block correlation, with ~90 % coverage at 90 % nominal.
- **ECE** uses 15 equal-mass bins. Predictions bunch near the 18 % base rate, so equal-width bins would leave most bins empty.
- **Volume anomaly** = |count − median| > k·1.4826·MAD (robust to the anomaly it is looking for).

## Check catalogue

| Code | Stage | Severity | Name | Why it exists |
|---|---|---|---|---|
| MLD001 | data | error | required-columns | Every downstream step assumes these columns; a renamed upstream column must fail here, not as a silent OOV. |
| MLD002 | data | error | binary-label | The label must be exactly {0,1} and both classes must occur. |
| MLD003 | data | error | unique-ids | Duplicate event ids double-count clicks and can straddle a split. |
| MLD004 | data | error | null-budget | Nulls in declared columns must stay within the configured budget. |
| MLD005 | data | error | time-parses | Every split, counter and drift window is keyed on event time; an unparseable timestamp silently drops rows. |
| MLD006 | data | warning | period-volume | A period with abnormal volume (outage, partial day, duplicate load) distorts per-period metrics and drift. |
| MLD007 | data | warning | dominant-values | A categorical where one value dominates is usually a placeholder (unknown device, default site); treating it as an identity merges unrelated entities. |
| MLD008 | data | error | foreign-keys | Events whose entity is missing get no entity features; a broken join must fail rather than become cold start. |
| MLD009 | data | error | event-after-creation | An event before its entity existed means a corrupted timestamp or an entity attribute recorded after the fact. |
| MLD010 | data | warning | constant-columns | A constant column carries no signal and usually signals a broken upstream join. |
| MLL001 | leakage | error | split-disjoint | An event in two splits is evaluated on data it was trained on. |
| MLL002 | leakage | error | split-temporal-order | Production predicts the future from the past; each split must start strictly after the previous one ends, and match the manifest windows. |
| MLL003 | leakage | error | fit-window | Every fitted object (vocabulary, encoder, prior, scaler, model, calibrator) must be fitted only on data that precedes the holdout, inside the split it declares. |
| MLL004 | leakage | error | shuffled-label-auc | Retraining on shuffled labels must give chance-level AUC; anything else means the pipeline leaks the label. |
| MLL005 | leakage | error | univariate-feature-auc | No single feature should predict clicks almost perfectly; one that does is usually derived from the label. |
| MLL006 | leakage | error | holdout-touched-once | Evaluating repeatedly on the holdout and picking the best turns it into a validation set; each model version gets one holdout evaluation. |
| MLL007 | leakage | warning | adversarial-validation | A classifier that separates train from holdout rows with high AUC means strong covariate shift; offline metrics on that holdout will not transfer. |
| MLM001 | model | error | predictions-contract | Every gate below is computed from predictions.parquet; it must hold primary and baseline predictions on every evaluation split for the same ids. |
| MLM002 | model | error | prediction-sanity | Probabilities must be finite, strictly inside (0,1) and not constant; a 0 or 1 makes log loss infinite and a constant model ranks nothing. |
| MLM003 | model | error | holdout-normalized-entropy | Normalized entropy (log loss / base-rate entropy) is the metric the auction consumes; it must clear the floor on the untouched holdout. |
| MLM004 | model | error | beats-baseline | The primary model must beat the baseline on the holdout with a paired, hour-block bootstrap CI that excludes zero; a point estimate alone can be noise. |
| MLM005 | model | error | calibration-ratio | Ranking by pCTR x bid needs calibrated probabilities; the mean prediction must match the observed holdout rate. |
| MLM006 | model | warning | calibration-per-period | Aggregate calibration can hide daily swings; each evaluation day must stay calibrated or drift needs an online correction. |
| MLM007 | model | error | expected-calibration-error | The ratio can be 1.0 while low and high scores are both wrong; ECE checks calibration across the score range. |
| MLM008 | model | error | required-slices | Averages hide cold-start and minority-segment failures; required slices must be present and populated. |
| MLM009 | model | warning | slice-normalized-entropy | A slice where the model is worse than predicting its own base rate is a segment the model actively hurts. |
| MLP001 | policy | warning | ope-effective-sample-size | An importance-weighted estimate resting on a few heavy weights is noise; effective sample size says how many rows really support it. |
| MLP002 | policy | error | ope-intervals | Policy comparisons need intervals; an estimate outside its own CI or a CI with no width is a reporting bug. |
| MLP003 | policy | error | decisions-respect-gates | A gated candidate (brand safety, frequency cap, budget) must never be served, whatever its score. |
| MLP004 | policy | error | logged-propensities | Future off-policy evaluation and unbiased retraining need the probability of every served ad, in (0, 1]; a lone eligible candidate must log 1. |
| MLR001 | repro | error | manifest-valid | Without data hash, code version, config hash, seeds and windows a model cannot be rebuilt or audited. |
| MLR002 | repro | error | data-hash-matches | Reported metrics are only meaningful for the exact bytes the model was trained on. |
| MLR003 | repro | error | clean-git-tree | A model trained from uncommitted code cannot be traced to a reviewable commit. |
| MLR004 | repro | error | seed-count | Neural nets vary across seeds by about as much as many feature changes; one seed cannot separate the two. |
| MLS001 | static | error | serving-no-training-deps | The serving path must not import training frameworks, even transitively: they bloat the image, slow cold starts and invite fitting code into the request path. |
| MLS002 | static | error | serving-uses-shared-features | Training and serving must build features with the same code; a serving path that re-implements features is the classic source of train/serve skew. |
| MLS003 | static | error | no-shuffled-split | CTR data is time-ordered; random or k-fold splits leak future hours, shared users and repeated creatives into training and overstate offline metrics. |
| MLS004 | static | error | no-global-rng | Global RNG state makes results depend on call order and imports; explicit generators make every stochastic step reproducible from the run's seed. |
| MLS005 | static | error | safe-model-loading | Model artifacts travel through buckets and registries; pickle-based loading executes code from them. |
| MLS006 | static | error | no-notebooks | Notebooks hide execution order and state; every result in the docs must come from a re-runnable command. |
| MLS007 | static | warning | no-fit-on-eval-data | Fitting anything on validation/test data leaks it. Legitimate cases (calibrator on validation) must be marked explicitly so a reviewer sees them. |
| MLV001 | serving | error | train-serve-parity | The serving path must produce the same features as the offline pipeline for the same rows; any difference is train/serve skew. |
| MLV002 | serving | error | onnx-parity | The exported model is what serves traffic; it must reproduce the trained model's outputs. |
| MLV003 | serving | error | latency-p99 | The ad slot waits for the ranker; p99 over budget means blank slots or a timed-out auction. |
| MLV004 | serving | error | load-error-rate | Fast responses do not count if they are errors. |
| MLX001 | drift | warning | feature-drift-psi | PSI above 0.25 marks a feature whose distribution moved enough to invalidate what the model learned about it. |

## Result on the shipped run (`uv run mlcheck .`)

**46 checks: 42 pass, 0 fail, 4 warn.** Every warning is a documented property of the data, not a defect:

| Warning | Finding | Where it is handled |
|---|---|---|
| MLD006 period-volume | 2014-10-30 has 22,956 rows (partial day) | Kept in test, never read alone ([01](01-data.md)) |
| MLD007 dominant-values | `device_id = a99f214a` on 82 %, plus four low-cardinality fields | User proxy and the `device_id_real` token ([01](01-data.md), [02](02-features.md)) |
| MLL007 adversarial-validation | Train vs test AUC 0.96, driven by creative and campaign rotation | Daily retraining, hierarchy backoff ([07](07-drift-adaptation.md)) |
| MLX001 feature-drift-psi | PSI > 0.25 on ad ids, `app_id`, and `hour_of_day` (partial last day) | Same as above |

Gates that pass with margin:
- leakage (shuffled-label AUC 0.472; strongest single feature 0.671);
- beats baseline, CI excluding zero;
- calibration 1.012 and ECE 0.005;
- exact train/serve parity, ONNX within 1.9e-6;
- p99 26 ms at 400 rps;
- 3,000 decisions respecting gates with valid propensities.

## Tests

`tools/mlcheck/tests`: a synthetic **golden project** (sources, raw data and a complete run) passes all 46 checks. One **breakage per check** proves each gate fails on the defect it targets, and a test asserts the breakage table covers every registered code. Further tests cover stats, CLI and edge cases (missing artifact, missing source, suppression, unconfigured section). Results: 111 tests, 94 % branch coverage; ruff (incl. bandit and pydocstyle rules) clean; pyright strict clean.
