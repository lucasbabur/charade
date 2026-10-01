# Plan — Simula CTR ranking take-home (v2, detailed)

**Index**
0. Data findings that decide the design
1. Principles (the "ML style")
2. Repository layout (every folder, what lives there, why)
3. Data layer: contracts, cleaning, edge cases
4. Split and leakage discipline
5. Features
6. Character text: embeddings and LLM enrichment (paid models)
7. Models
8. Evaluation protocol
9. Candidate ranking policy
10. Cold start
11. Drift and adaptation
12. Serving (<50 ms p99)
13. Observability and propensity logging
14. Infrastructure (AWS, Terraform, Docker)
15. Quality checks (every check, threshold and where it runs)
16. Tests (what each suite proves)
17. CI/CD
18. Docs set
19. Commit sequence
20. Time budget, risks, and the reasoning points for the recording

---

## 0. Data findings that decide the design

DuckDB profile of the provided CSVs, 2026-10-01.

| Finding | Number | Design consequence |
|---|---|---|
| Size and base rate | 1,000,000 rows, CTR 18.04 %, 222 hours, 2014-10-21 00h to 10-30 05h | Avazu sample plus a synthetic character layer. Day-grained temporal split. |
| Daily CTR regime | 16.5 % (Wed 22, Tue 28) to 19.6 % (Mon 27). Test days 29–30 are low (17.3 %, 16.6 %) | Calibration drifts between days. Report calibration ratio per day and correct for it online. |
| `device_id` placeholder | `a99f214a` on 82 % of rows | It is not a user. User proxy = `device_id` when real, otherwise `hash(device_ip, device_model)`. |
| One-shot IPs | 441k of 547k `device_ip` values appear once | Most users are cold by construction. No user-ID embeddings; behaviour counters only. |
| Ad hierarchy in the C-features | C14→C15, C16, C17, C21 deterministic (1.000); C17→C21 1.000; C14→C20 0.30; app_id→C14 0.41 | C14 = creative, C17 = campaign, C21 = advertiser, C15×C16 = creative size. **Candidate = `banner_pos` + C14 (+ its derived C15–C19, C21).** C20 and C1 vary within a creative, so they are context features, not candidate features. 13,322 distinct C14–C21 tuples. |
| Surface | app traffic (`site_id=85f751fd`) 13.4 % vs site 20.6 % | Surface type is a top feature and a monitoring slice. |
| Position | `banner_pos` 0: 17.4 %, 1: 19.6 %, 7: 35 % (1.1k rows) | Position is part of the candidate. Rank on position-debiased quality (§9). |
| Strong low-cardinality effects | `device_conn_type=3` 6.5 %; `C1` ∈ {1001, 1007} ~4 % | Cheap wins; the GBDT will find them. The DCN gets them as embeddings. |
| Genre (name prefix) | romance 22.5 %, horror 21.8 %, mentor 14.3 %, the rest ~17.3 % | Strongest cold-start signal, available before the first click. |
| Safety tier | mature +2–3 pp over sfw/suggestive on every day | Real signal and also the brand-safety gating dimension. |
| Character beyond genre | Per-character residual sd 0.019 against a binomial noise floor of ~0.016 → true sd ≈ 0.010 | Character identity adds about 1 pp beyond genre. It is small but real, which justifies shrinkage instead of raw IDs. |
| Description words | Largest within-genre word lift is 0.44 pp; 4,548 unique descriptions from obvious templates | The text main effect is weak. Text should earn its place through **character × ad interactions** and cold start. Pre-registered expectation: small lift. Measure it rather than assume it. |
| Conversation features | CTR 17.9–18.3 % across every `conversation_turn` and `session_msg_count` bucket | Probably noise. Keep only if ablation shows lift. `session_msg_count` is in the serving contract, but "total messages in session" leaks if it counts future messages, so the docs flag it. |
| `num_interactions` | Quintiles flat at 17.9–18.4 % | Weak, and it is a snapshot (possible future leak). Log-bucket it and ablate it. |
| Character cold start | 4,997 characters; median 64 impressions, p10 = 11; 300–1,500 first seen per day; 243 created after 10-20; 0 impressions before `created_at` | A natural cold slice exists in the test window. Foreign-key and time integrity are clean. |
| Hygiene | No nulls, no duplicate ids, hours valid, all rows 27 columns | Validation is still enforced; the serving input is untrusted. |

---

## 1. Principles (the "ML style")

1. **Baseline first; complexity has to earn its place.** Every component beyond the logistic baseline ships only if a paired-bootstrap CI on validation logloss excludes zero. Ablations are a deliverable, not an afterthought.
2. **Pre-registration.** The metric, split, decision rules and expected effects are written in `docs/adr/` before results exist. The test window is touched once; the run id is recorded.
3. **Time is the only honest split.** Nothing fitted (vocabularies, encoders, priors, PCA, calibrators, scalers) sees data after the train cutoff. Counters use strictly earlier hours (order within an hour is unknown, so the same hour is excluded).
4. **One transform, two callers.** Training and the API call the same `cameo.features` functions (mlcheck MLS002). A parity report guards against train/serve skew (MLV001).
5. **Calibration is first-class.** Ranking uses pCTR × bid, so probabilities must be right, not just well ordered. Logloss and normalized entropy are primary; AUC is secondary.
6. **Uncertainty in every number.** Block-bootstrap CIs (hour blocks), 3 seeds for neural models, effective sample size for off-policy estimates.
7. **Slices before averages.** Cold/warm character, genre, tier, surface, day and new/seen user are reported for every model.
8. **Reproducible from raw.** `make reproduce` goes from CSV to every table and figure in the docs. Seeds are fixed, torch deterministic flags are on, and each artifact has a manifest (data sha256, git sha, config hash, train window, metrics).
9. **Experiments are config tables, not notebooks.** `[tool.cameo.experiments.<name>]` in pyproject.toml → CLI → MLflow (local file store) and `reports/`. There are no notebooks in the repo.
10. **Pure math, thin I/O.** Feature, ranking, evaluation and drift math are pure numpy/polars functions, unit-tested without I/O. Torch, ONNX, Redis and API clients stay in `models/`, `serving/` and `text/`. The serving import boundary is enforced by mlcheck MLS001.

Code style: Python 3.12, full type hints, pyright strict on `app/`, Google docstrings on public functions, pydantic v2 at every boundary, structlog JSON logging (no f-strings in log calls), no bare `except`, domain exceptions in `shared/errors.py`.

---

## 2. Repository layout

Organised by ML concern, not by clean-architecture layer. Those layers protect against swapping databases and UIs. The risks here are train/serve skew, leakage, a heavy serving image and irreproducible results, so the enforced rules target those (mlcheck MLS001–007, §15).

```
take_home_assigment/
├── src/cameo/
│   ├── config.py            # pydantic-settings reading [tool.cameo] from pyproject.toml; env overrides; secrets from env only
│   ├── data/                # pandera contracts, polars loader (CSV → day-partitioned parquet), temporal split, backtest folds
│   ├── features/            # THE transform: user proxy, causal counters, vocabularies, character features; used by training and serving
│   ├── text/                # embedding clients (Gemini, Voyage, OpenAI, local Qwen), Claude attribute extraction, sha256 cache
│   ├── models/              # baselines, LightGBM, DCN-v2 (torch), trainer, calibration, ONNX export, propensity model
│   ├── evaluation/          # metrics, hour-block bootstrap, slices, ablations, OPE, artifact writers (mlcheck contract)
│   ├── ranking/             # gates, EV, fatigue, pacing, Thompson exploration + propensity, policy
│   ├── coldstart/           # beta-binomial priors, graduation analysis
│   ├── drift/               # PSI, churn, novelty, adaptation simulator
│   ├── serving/             # FastAPI app, routes, schemas, Redis store, ONNX scorer, observability; imports features + ranking only
│   └── cli.py               # typer: train | evaluate | drift | simulate | export | serve | sample
├── tools/mlcheck/           # ML release gates (workspace package, own tests) — docs/mlcheck.md
├── tests/                   # unit/<concern>/, integration/, api/, property/, load/, fixtures/ (20k-row sample)
├── artifacts/current/       # gitignored run output = mlcheck artifact contract
├── reports/                 # generated, committed: figures, tables, sample_rankings.json
├── docker/                  # api.Dockerfile (onnxruntime, no torch), train.Dockerfile
├── docker-compose.yml       # api + redis + prometheus
├── infra/terraform/         # modules/ + envs/{staging,prod}
├── .github/workflows/       # ci.yml, cd.yml
├── docs/
└── pyproject.toml           # the only settings file: workspace, [tool.cameo], [tool.mlcheck], ruff, pyright, pytest, coverage
```

Import rules (enforced by mlcheck, not by convention):
- `serving` never reaches torch, lightgbm, sklearn, optuna or mlflow, even transitively (MLS001).
- `serving` must import `features` (MLS002).
- No shuffled splits (MLS003), no global RNG (MLS004), no pickle or unrestricted `torch.load` (MLS005), no notebooks (MLS006), no unmarked fit on evaluation data (MLS007).

---

## 3. Data layer: contracts, cleaning, edge cases

- **Ingest:** polars reads the CSV with an explicit schema (ids as `Utf8`, not int: `id` exceeds int64 range semantics, and hashes have leading zeros). Writes partitioned parquet by day to `artifacts/cache/`.
- **pandera contract (offline, fails the run):** `hour` matches `^\d{8}$` with HH ≤ 23; `click` ∈ {0,1}; `1 ≤ conversation_turn ≤ session_msg_count`; `banner_pos` ∈ known set; `character_id` foreign key exists; impression time ≥ `created_at`; unique `id`; `safety_tier` ∈ {sfw, suggestive, mature}; `creator_type` ∈ {community, official}; descriptions non-empty (quoted commas parse correctly).
- **Soft checks (warn plus a metric, no failure):** share of the placeholder `device_id`, OOV rate per categorical against the train vocabulary, hourly row count outside ±4σ.
- **Serving edge cases (pydantic → 422 unless noted):**

| Case | Behaviour |
|---|---|
| Unknown `character_id` | Cold path (metadata features); `cold_start=true` in the response; metric |
| `character_id` missing from the request | 422 (part of the contract) |
| Unknown categorical value | OOV index; metric `oov_total{feature}` |
| `conversation_turn > session_msg_count` | Clamp; metric `input_repaired_total` |
| Empty candidate list | 422 |
| N > 500 candidates | 422 (protects latency) |
| Duplicate candidate ids | Dedupe, keep the first; warn in the response |
| Every candidate gated | 200, empty `ranked`, `reason="all_gated"` |
| Redis timeout (5 ms) or error | Default counters, `degraded=true`, metric; never a 5xx |
| Missing character embedding | Genre-mean embedding fallback |
| NaN/inf score | Drop the candidate, `score_error_total`; if all are dropped → 503 |
| Model not loaded | `/ready` 503; the ALB keeps the task out of rotation |
| Ties | Stable sort by score, then candidate id (deterministic) |
| Malformed hour / timezone | Accept ISO-8601 or `YYMMDDHH`; normalize to UTC |
| Body > 256 KB | 413 from middleware |

---

## 4. Split and leakage discipline

- **Holdout:** train on 10-21 → 10-27 (7 days, each weekday once); validation 10-28 (early stopping, hyperparameters, calibrator, ablation decisions); test 10-29 + 10-30 (partial), touched once.
- **Final model:** refit on 21–28 with the epoch count chosen on validation; evaluate on test.
- **Rolling-origin backtest:** train ≤ d, test d+1, for d = 24…29. It produces the staleness curve (frozen model against daily retrain) for §11.
- **Why not random:** random splits leak hour-level effects, shared devices and repeated creatives across folds, which inflates AUC and hides drift and cold start. Production is "train on the past, serve the future".
- **Leakage controls:**
  - Every fitted object is fitted on the train window only; the test asserts fit timestamps ≤ cutoff.
  - Counters are computed over hours strictly before the impression hour.
  - Target-derived character features (empirical-Bayes CTR) are time-respecting: for training rows they use only earlier days (expanding window); for validation and test they use the full train window.
  - `num_interactions` and `session_msg_count` are snapshot or "total" fields → ablated and flagged.
  - Label-shuffle test: a model trained on shuffled labels must give validation AUC ∈ [0.48, 0.52].
  - Adversarial validation (train against test classifier) runs as a drift diagnostic. A feature with AUC > 0.8 on its own is investigated before use.

---

## 5. Features

| Group | Features | Encoding |
|---|---|---|
| Time | hour-of-day, day-of-week | Embedding plus sin/cos. Day-of-week is ablated (one instance of each in train) |
| Publisher | `site_id`, `site_domain`, `site_category`, `app_id`, `app_domain`, `app_category`, surface type (app/site) | Vocabulary, min-count 5 → OOV; log train frequency |
| Device | `device_model`, `device_type`, `device_conn_type`, `device_id` only when not the placeholder | Vocabulary + OOV |
| Context-side C | `C1`, `C20` | Vocabulary |
| Candidate (ad) | `banner_pos`, `C14` (creative), `C17` (campaign), `C21` (advertiser), `C15×C16` (size), `C18`, `C19` | Vocabulary. Hierarchical backoff: an unseen C14 still has C17/C21 |
| User-proxy counters | Impressions in the previous 1 h / 24 h; impressions of this C17 to this user in the previous 24 h (fatigue); hours since last impression; user seen-before flag | log1p; identical definitions online in Redis |
| Character metadata | genre, `safety_tier`, `creator_type`, log-bucket `num_interactions`, character age in days (impression − `created_at`), character-is-new (< 3 days) | Embeddings / buckets |
| Character text | Template slots (§6), LLM attributes (§6), description embedding reduced to 32 dims | Categorical + dense |
| Character identity | `character_id` embedding, min-count 20 → "rare" bucket, **10 % ID dropout in training** so the model learns the metadata fallback | Embedding |
| Character prior | Empirical-Bayes CTR shrunk toward the genre×tier prior; log impression count | Dense (LightGBM; also tried in the DCN) |
| Conversation | log `conversation_turn`, log `session_msg_count`, turn/session ratio, first-turn flag | Behind a flag; kept only if the ablation shows lift |
| Explicit crosses | genre × C17, safety_tier × C21, surface × banner_pos | For the logistic and GBDT models; the DCN learns crosses itself |

---

## 6. Character text: embeddings and LLM enrichment

**Observation:** the descriptions come from templates (`"A {adj} {role} who …, {adj} {clause}"`), and words add ≤ 0.44 pp within a genre. In production the descriptions will be free text written by creators. So the plan builds three layers and lets the validation set decide:

1. **Template slot parser (deterministic, free):** regex per genre template → `role`, `adj_1`, `adj_2`, `setting`. This is the exact upper bound of what text can give on this synthetic data. It is documented as a dataset artifact that will not survive real free text.
2. **Embeddings (production path):** computed offline, never on the hot path. The total corpus is about 5k × 30 tokens ≈ 150k tokens, so every paid model costs under $0.05 → **choose on quality, not price.** Bake-off:

| Candidate | Why | Settings |
|---|---|---|
| **Google `gemini-embedding-001`** (default hypothesis) | Leads API models on MTEB classification and clustering (+9.6 / +3.7 over the runner-up in the release paper); it has `task_type=CLASSIFICATION`/`CLUSTERING` | 768-d MRL; batch API |
| Voyage `voyage-4-large` (Jan 2026) | Strongest current general-purpose API family; MRL 256–2048 | 1024-d, `input_type=document` |
| OpenAI `text-embedding-3-large` | Widely used reference point | `dimensions=1024` |
| Qwen3-Embedding-0.6B (local, GTX 1660 6 GB) | Open-weights control; proves the pipeline runs without keys | sentence-transformers, fp32 |

   Selection protocol (written down before results):
   - (a) Intrinsic sanity: kNN accuracy predicting genre and tier from the embedding (must be > 0.9 or the embedding is broken).
   - (b) Linear probe: predict the character EB-shrunk CTR *residual* over genre×tier, train characters → validation characters, Spearman ρ with a CI.
   - (c) Extrinsic, decisive: DCN validation logloss overall and on the **cold-character slice**, with a paired bootstrap against the no-text model.
   - The winner by (c) ships. Ties go to the cheapest, lowest-dimension option. If none beats the slot parser + LLM attributes, ship no embedding and say so.
   - Dimensionality: PCA to 32 fitted on train-window characters (stored in the artifact). MRL truncation is compared.
3. **LLM structured attributes (Claude Haiku 4.5, `claude-haiku-4-5-20251001`, structured JSON output, temperature 0):**
   - Fields: `archetype`, `tone`, `romance_level` 0–3, `dominance` 0–3, `violence_level` 0–3, `humor` 0–3, `nsfw_risk` 0–3, `target_audience`.
   - Uses: interpretable categorical features that transfer to free text; a **safety cross-check** (`nsfw_risk` against `safety_tier` disagreement → flags mislabelled characters for brand-safety review); readable ranking justifications.
   - Cost under $1 for 5k characters. Cached by sha256(description) + prompt version.

**Operational rules:**
- The `EmbeddingProvider` and `AttributeExtractor` contracts have retries (tenacity, exponential backoff), rate limits and idempotent caching.
- API keys come from env locally and from AWS Secrets Manager in the enrichment job.
- CI never calls paid APIs. Clients are tested with recorded responses (respx/pytest-recording). The winning embeddings and attributes are committed as float16 parquet, so reviewers reproduce everything without keys.
- New characters: a publish event → enrichment worker → feature-store row in minutes. Until then the API uses the genre-mean embedding and default attributes.

---

## 7. Models

| # | Model | Role | Details |
|---|---|---|---|
| M0 | Global prior; hour × surface prior | Sanity floor | NE = 1.0 by definition for the global prior |
| M1 | Logistic regression on hashed one-hots + explicit crosses | Classic baseline | sklearn `SGDClassifier(log_loss)`, 2^22 hashing, L2 tuned on validation |
| M2 | LightGBM | Strong tabular challenger, offline reference | Native categoricals, count/EB features, early stopping on validation logloss, `num_leaves`/`min_data_in_leaf`/`learning_rate` via Optuna (50 trials, validation only) |
| M3 | **DCN-v2 (PyTorch)** | Ships to serving | Per-field embeddings (dim = min(16, ⌈6·card^0.25⌉)), 3 low-rank cross layers (rank 64), deep tower 256-128 with ReLU and dropout 0.1, stacked; dense inputs (text PCA, counters) batch-normalized; BCE loss, AdamW lr 1e-3 / wd 1e-5, batch 4096, early stopping patience 2 on validation logloss; 3 seeds |
| M4 | Calibration | On top of M2/M3 | Isotonic on validation (Platt if isotonic overfits per CI); per-day calibration ratio monitored |
| M5 | Propensity model | Off-policy evaluation | Multinomial LightGBM P(C17 \| context) as the logging-policy estimate (campaign level keeps the class count manageable); clipped |

- **Why DCN-v2 as the serving model:** ranking is a question of character/context × ad crosses; the cross network learns them explicitly, exports cleanly to ONNX, and scores 100 candidates in one batched forward pass on CPU in about 1 ms. LightGBM stays as the honest yardstick.
- **Decision rule (pre-registered):** ship M3 if its test logloss is within the CI of M2 or better. If M2 clearly wins, ship M2 (also served through ONNX via `onnxmltools`) and document what the DCN lacked.
- **Ablations (one table, validation logloss Δ with CIs, 3 seeds):**
  - remove character metadata
  - remove character ID
  - remove the ID dropout
  - remove the text features
  - each embedding provider in turn
  - remove the LLM attributes
  - remove the conversation features
  - remove the user counters
  - remove `num_interactions`
  - remove the EB prior
  - remove the day-of-week feature
- Training runs on CPU (24 cores) by default; the GPU is optional. The 1M rows train in minutes. Data loading uses pre-tensorized parquet, not a per-row Dataset.

---

## 8. Evaluation protocol

- **Primary metrics:** logloss and **normalized entropy** (logloss ÷ base-rate entropy). They match what the auction consumes, pCTR × bid.
- **Secondary metrics:**
  - ROC-AUC and PR-AUC
  - calibration: ECE (15 equal-mass bins), predicted/observed ratio overall and per day and slice, reliability diagrams
  - **Group AUC** within (user proxy, hour), which is closer to "rank ads for this person now"
- **Uncertainty:** hour-block bootstrap (1,000 resamples) for every metric; a **paired** bootstrap for model differences.
- **Slices** (every metric):
  - character train support: 0 / 1–20 / 21–200 / 200+
  - genre, safety tier, creator type, surface, `banner_pos`, day
  - user proxy new vs seen; placeholder vs real `device_id`
- **Ranking quality (logged data shows one ad per impression):**
  - **SNIPS** and **doubly robust** estimates of the policy's CTR using the M5 propensities, weight clipping at 10, effective sample size reported; plus a replay estimate on matching decisions.
  - These are labelled as estimates with assumptions: no unobserved confounding beyond the context features; and the propensity model, not the true logging policy.
- **Operating checks:**
  - scoring latency per batch of 100 (pytest-benchmark)
  - model size
  - ONNX/torch max abs difference
- **Reporting:** `reports/tables/*.md` generated, MLflow run ids linked, figures in `reports/figures/`.

---

## 9. Candidate ranking policy

Input: the opportunity context plus N candidates (`candidate_id`, `banner_pos`, C14 and derived C-fields, optional `bid`, `advertiser_id`=C21, `campaign_id`=C17).

1. **Hard gates** (each rejection returns a `gate_reason`):
   - **Brand safety:** matrix of advertiser allowed tiers × character `safety_tier` (`[tool.cameo.policy.safety_matrix]`; default: mature characters only receive advertisers allowlisted for mature; unknown advertiser → sfw-only). The LLM `nsfw_risk` disagreement upgrades the tier, conservatively.
   - **Frequency cap:** ≤ K impressions of the same campaign to the same user proxy in 24 h (K = 3 default).
   - **Budget exhausted.**
2. **Score:** one batched ONNX forward pass → calibrated pCTR.
3. **Expected value:** `pCTR × bid × pacing_multiplier`. The bid defaults to 1, which reduces to pure CTR ranking as the brief asks. Pacing comes from a per-campaign PI controller on spend against a linear target curve. In the prototype, state lives in Redis and is updated asynchronously.
4. **Fatigue:** a multiplicative `exp(−λ · exposures_24h(user, campaign))` and a cohort-level penalty from §11. λ is fitted from the data: the CTR decay against prior exposures of the same C17 to the same user proxy.
5. **Uncertainty and exploration:**
   - Each candidate's posterior is Beta(pCTR·n_eff, (1−pCTR)·n_eff), where n_eff is the training support of (C17, genre) capped at 1,000. Unseen pairs get small n_eff, so wide posteriors.
   - **Greedy on 95 % of traffic; Thompson sampling on a 5 % exploration bucket** (deterministic by hash(request_id)).
   - Propensity of the shown ad = 0.95·1[greedy pick] + 0.05·P_TS(pick), where P_TS comes from a 64-draw Monte Carlo (≈0.2 ms for N=100).
   - Why: exploration spend is bounded and auditable, and every decision becomes usable for off-policy evaluation and unbiased retraining later.
6. **Position:** `banner_pos` is a model input. When candidates differ in slot, rank on a debiased quality score (pCTR at reference position 0) for ad quality, and on the slot-specific pCTR for expected value. Both are returned.
7. **Response per candidate:** `rank`, `pctr`, `pctr_ci` (posterior 5–95 %), `expected_value`, `explored`, `propensity`, `gate_reasons`, `cold_start_flags`, and top-3 feature contributions (gradient × input on the dense part; cheap). The response also carries `model_version`, `policy_version`, `degraded` and `request_id`.
8. **Justification when uncertain:**
   - If the top two candidates' CIs overlap and the leader has low support, the response says so (`decision_confidence: low`).
   - The policy prefers the candidate with higher support at equal EV, a conservative tie-break that saves exploration for the explicit bucket.

---

## 10. Cold start

| Entity | Signal before any click | Mechanism |
|---|---|---|
| New character | genre, safety tier, creator type, age, template slots / LLM attributes, description embedding, `num_interactions` bucket | "Rare" ID bucket plus the metadata path, trained for this via 10 % ID dropout; the EB prior from genre×tier |
| New user (proxy) | context (site/app, hour), device model/type/connection, C1/C20, the character they are chatting with | No user-ID features by design; counters start at zero with a "seen-before=0" flag |
| New device | device model/type/connection, IP-level counters if the IP was seen | Same as new user |
| New creative (C14) | campaign (C17), advertiser (C21), size, position | Hierarchical backoff in the vocabulary; the exploration bucket gives it traffic |

- **Bootstrapping:** a beta-binomial prior per genre×tier, fitted by method of moments on train (α, β), so a character's estimate is `(clicks+α)/(n+α+β)`. Thompson exploration (§9) routes a bounded share of traffic to uncertain entities.
- **Graduation rule** (evidence-based, not a magic number):
  - An entity graduates when its own evidence dominates the prior, `n ≥ α + β`, **and** the learning curve says the ID path beats metadata-only.
  - The learning curve: for warm characters, logloss on impressions k+1… using metadata-only against the ID model, as a function of k. The crossover k* sets the vocabulary min-count at retrain.
  - It is reported per genre, since priors differ in strength.
- **Evaluation:** characters first seen in test (and new user proxies) are a dedicated slice. Success criterion: cold-slice NE within 2 % of warm-slice NE. The gap is reported either way.
- **Permutation importance on the cold slice** answers "which features carry signal before clicks".

---

## 11. Drift and adaptation

**Analysis** (`reports/figures/drift_*.png`, numbers in `docs/06-drift.md`):
- CTR by hour and by day, overall and by genre, tier, surface and `banner_pos`.
- PSI per feature per day against the train window (alert threshold 0.2); adversarial-validation AUC train↔test and its top features.
- Character mix: share of impressions from first-seen characters per day, Jaccard of the top-100 characters day over day, HHI of character share, and the rank turnover of the top genres.
- Novelty: CTR against character age (days since `created_at`) and against the cumulative impressions already served (wear-out).
- Ad rotation: C14 birth and death per day and the share of impressions on creatives unseen in train.
- Staleness cost: rolling-origin backtest, frozen day-24 model against daily retrain, logloss and calibration ratio per day.

**Adaptation prototype** (`drift/adaptation.py`, replay simulator over test days):
- Arms = (cohort = genre×tier) × campaign. Discounted Thompson sampling (γ ∈ {0.9, 0.97, 0.99} per hour) with the model pCTR as the prior mean.
- Fatigue constraint: when a dominant cohort's exposure share to one campaign exceeds the cap within a rolling window, its score is penalized (Lagrangian-style multiplier updated hourly).
- Online calibration: exponentially weighted correction of the global calibration ratio from recent clicks.
- **Output:** a Pareto curve of SNIPS/replay CTR against exposure concentration (HHI, repeat-exposure rate, dominant-cohort share). Policies compared: greedy, TS, TS+fatigue.
- **Target:** TS+fatigue CTR within the CI of greedy at materially lower concentration. Every replay estimate is reported with its effective sample size and an honest caveat about replay bias.

---

## 12. Serving (<50 ms p99)

**Request path** (`POST /v1/rank`):

| Stage | Budget p50 | Implementation |
|---|---|---|
| Parse and validate | 0.5 ms | pydantic v2, orjson response class |
| Feature fetch | 2–3 ms | **One pipelined Redis round trip** (user counters, cohort TS stats, pacing state); 5 ms timeout → defaults, `degraded` |
| In-process lookups | 0.3 ms | Character table + PCA embeddings + attributes + vocabularies + priors in memory (5k characters ≈ 1 MB) |
| Feature assembly | 1 ms | Vectorized numpy over N candidates; context features computed once, broadcast |
| Inference | 1–3 ms (N ≤ 200) | ONNX Runtime CPU, one batched call, `intra_op_num_threads=1` per worker, session warmed at startup |
| Policy | 0.5 ms | numpy gates, EV, TS draws |
| Async log | ~0 | Decision log to a bounded in-memory queue → background task → stdout JSON / Firehose |
| **Total** | **~8 ms p50; target p99 < 25 ms in-container** | Leaves headroom for the network and ALB within 50 ms |

- **Runtime:** uvicorn with one worker per vCPU (gunicorn manager); `async def` routes with redis.asyncio. Inference runs inline because it is CPU-bound and ~2 ms, cheaper than a threadpool hop. The API image has no torch (ONNX only), so it is small and starts fast.
- **Precomputed:** character features and embeddings (refreshed by the enrichment job; hot-reloaded by version), vocabularies, the calibration map as a lookup table, priors, hour features.
- **Approximations:** PCA-reduced embeddings; counters in hour buckets (Redis hash, TTL 25 h) instead of exact sliding windows; TS posteriors refreshed every 5 min, not per click; int8 ONNX quantization only if measurements demand it (measured, not assumed).
- **Model rollout:** artifacts are versioned in S3; containers pull a pinned version; a shadow mode flag scores with model B and logs it without serving it; a canary percentage by request hash.
- **Proof:** locust against docker compose (api + redis), 200 RPS for 2 min at N=100 candidates. p50/p95/p99 are written to `docs/07-serving.md`; CI runs a 30 s smoke version that asserts p99 < 50 ms.
- **Endpoints:** `POST /v1/rank`, `POST /v1/predict` (single impression pCTR), `GET /health` (liveness), `GET /ready` (artifacts loaded, Redis reachable or degraded-OK), `GET /metrics` (Prometheus), `GET /v1/model` (versions and manifest).

---

## 13. Observability and propensity logging

- **Metrics (Prometheus):**
  - `rank_latency_seconds{stage}` histogram
  - `pctr` histogram (served distribution against offline → drift signal)
  - `cold_start_total{entity}`
  - `oov_total{feature}`
  - `gate_rejections_total{reason}`
  - `exploration_total`
  - `degraded_total{cause}`
  - `score_error_total`
  - `candidates_per_request` histogram
  - `model_info{version}` gauge
- **Traces:** OpenTelemetry spans per stage (FastAPI + Redis instrumentation), OTLP exporter (X-Ray / any collector).
- **Logs:** structlog JSON with `request_id`, versions and degraded flags; no PII. `device_ip` is hashed before logging.
- **Decision log** (the training and OPE data of tomorrow): `request_id`, timestamp, context features, every candidate's features/pCTR/EV/gate reason, the chosen candidate, its **propensity**, the exploration flag, model/policy versions. Clicks are joined by `request_id` in the warehouse.
- **Offline monitors** (scheduled job): feature PSI against the training snapshot; calibration ratio (predicted against observed, hourly, once clicks join); cold-start share; exploration CTR against greedy CTR.
- **Alarms** (CloudWatch):
  - ALB p99 > 40 ms for 5 min
  - 5xx > 0.5 %
  - degraded > 2 %
  - calibration ratio outside [0.9, 1.1] for 3 h
  - PSI > 0.25 on any top-10 feature

---

## 14. Infrastructure (AWS, Terraform, Docker)

**Docker**
- `api.Dockerfile`: uv multi-stage, `python:3.12-slim`, non-root uid 10001, read-only filesystem compatible, `HEALTHCHECK` on `/health`, artifacts pulled at start (or baked for the local demo).
- `train.Dockerfile`: torch CPU and training dependencies.
- `docker-compose.yml`: api, redis, prometheus.

**Terraform (validated, never applied)**

| Module | Resources |
|---|---|
| `network` | VPC across 3 AZs, private/public subnets, NAT, VPC endpoints for S3/ECR/Logs |
| `ecr` | Repositories api/train, scan on push, lifecycle keeps the last 20 |
| `ecs_api` | ECS Fargate service (2 vCPU / 4 GB, min 3 tasks across AZs), ALB + target group on `/ready`, autoscaling on CPU 50 % and ALB request count per target, deployment circuit breaker with rollback |
| `redis` | ElastiCache Redis 7, primary + replica, multi-AZ, encryption in transit and at rest, in private subnets |
| `artifacts` | S3 versioned bucket for models and features, SSE-KMS, public access block |
| `decision_logs` | Kinesis Firehose → S3 (parquet, partitioned by hour), lifecycle to IA after 30 d |
| `training_job` | ECS scheduled task (EventBridge, daily) for retrain + evaluate + gate + promote; enrichment task for new characters |
| `observability` | CloudWatch log groups (retention 30 d), alarms from §13, dashboard |
| `ci_oidc` | GitHub OIDC provider + least-privilege deploy role |
| secrets | Secrets Manager entries for embedding and Claude keys (enrichment task role only) |

- Environments: `envs/staging` and `envs/prod` compose the modules with different sizes; S3 backend with lock as a partial config.
- Checks:
  - `terraform fmt -check -recursive`
  - `terraform init -backend=false && terraform validate` per environment
  - `tflint` (aws ruleset)
  - `checkov` (or `trivy config`), with documented suppressions only

---

## 15. Quality checks

Two families. Generic Python quality comes from established tools. ML correctness comes from **mlcheck** (`tools/mlcheck`, 46 gates, catalogue and artifact contract in `docs/mlcheck.md`).

| Check | Tool | Gate | Runs in |
|---|---|---|---|
| Format, lint | ruff (E, W, F, I, B, BLE, UP, SIM, N, S, PL, PT, PERF, RUF, C90 ≤ 10, D google) | Zero findings | pre-commit, CI |
| Types | pyright strict on `src/` and `tools/` | Zero errors | pre-commit, CI |
| Layout-independent pycheck | `pycheck --only RF RL CX DC EG FL HI NA RA UD VU` + ED OQ (API docs) | Zero errors | CI |
| ML static gates | `mlcheck --stage static` | Zero FAIL | pre-commit, CI |
| Data contract | `mlcheck --stage data` + pandera in the pipeline | Zero FAIL; WARNs documented | CI (fixture), `make reproduce` (full) |
| Run gates | `mlcheck --stage repro leakage model serving policy drift --strict` | Zero FAIL; WARNs must be explained in docs | `make reproduce`, model promotion |
| Tests + coverage | pytest, pytest-cov | ≥ 85 % branch on src | CI |
| Property tests | hypothesis | All pass | CI |
| Security | pip-audit, trufflehog, semgrep | No high/critical | CI + nightly |
| Containers | hadolint, trivy | No high/critical with a fix | CI |
| Terraform | fmt, validate (both environments), tflint, checkov | All pass | CI |
| Commits | conventional-commit regex | Every message | CI |

Dropped: pycheck's layer checks (CA, FS, TM, SL, UC, DP, PN, CC, MC, CD, CI). They encode a business-app architecture this project does not use.

---

## 16. Tests (what each suite proves)

- **unit/<concern>:**
  - genre and slot parsers on every template
  - user proxy (placeholder detection)
  - counters are causal (a row at hour h never sees hour ≥ h)
  - vocabulary OOV handling
  - EB shrinkage matches the closed form; the prior fit recovers known α, β on synthetic data
  - calibration map is monotone
  - metrics against sklearn reference values
  - hour-block bootstrap CI covers the truth in ≈95 % of simulations
  - SNIPS/DR unbiased on a synthetic logging policy with known truth
  - PSI on known distributions
  - pacing controller converges on a synthetic spend curve
  - TS propensity Monte Carlo against the analytic value for 2 arms
- **unit/ranking + serving:** policy with an in-memory store and a stub scorer returning fixed pCTR — ordering, degraded paths, all-gated, dedupe.
- **property (hypothesis):**
  - ranking never returns a gated candidate
  - at equal bid and no fatigue, order is monotone in pCTR
  - same input + same request_id → identical output
  - propensities ∈ (0, 1] and the chosen ad's propensity > 0
  - the feature transform never raises on any string in any categorical field
  - output length ≤ input length
- **integration/infrastructure:**
  - polars gateway on the fixture CSV (schema, parquet roundtrip)
  - Redis gateway against a real Redis (testcontainers) including the timeout path
  - embedding/Claude clients against recorded HTTP cassettes (retries, rate limit, cache hits)
  - ONNX exporter → scorer roundtrip
- **api:** every edge case in §3 → status code and body; the OpenAPI schema validates; `/ready` flips with artifact presence; a request_id is propagated.
- **ml:**
  - (1) label-shuffle leakage test
  - (2) every fitted artifact's train-window end ≤ cutoff
  - (3) train/serve parity: 1,000 fixture rows through the offline pipeline vs. API feature assembly → identical tensors
  - (4) ONNX vs torch max abs diff < 1e-5
  - (5) quality floor on the fixture: NE < 0.98 and validation calibration ratio ∈ [0.9, 1.1]
  - (6) determinism: same seed → same metrics to 1e-6
  - (7) cold-character slice evaluation runs and is non-empty
- **load:** locustfile with realistic request generation sampled from the test day (cold characters included).

---

## 17. CI/CD

**`ci.yml` (PR + push to main), parallel jobs, uv cache:**
1. `lint`: ruff format/check, pyright, pycheck (layout-independent codes), `mlcheck --stage static`.
   `mlcheck-tests`: the mlcheck suite itself.
2. `test`: unit + property + api + integration (Redis service container), coverage gate, JUnit and coverage artifacts.
3. `ml-gates`: build features → train a tiny DCN on the fixture → calibrate → export ONNX → `tests/ml`. Uploads the model artifact.
4. `security`: pycheck --security, pip-audit, trufflehog.
5. `docker`: hadolint → build the api image (with the `ml-gates` artifact) → trivy → compose up → `/ready` → locust 30 s headless with a p99 assertion.
6. `terraform`: fmt, validate (both environments), tflint, checkov.

**`cd.yml` (tag `v*`):**
- OIDC → build and push to ECR → `terraform plan` artifact for staging → manual approval environment → apply + ECS deploy (circuit breaker rollback) → smoke.
- Model promotion is separate: the training job writes the candidate model; promotion requires its gate report (NE not worse than production by more than the CI, calibration ratio in band, latency benchmark passed) → the S3 "production" pointer is updated → tasks hot-reload.
- Documented; not executed (no AWS account).

**Nightly:** security profile, and full `make reproduce` on a self-hosted or manual run (the CSVs are not in git).

---

## 18. Docs set (`docs/`, dense, numbers only from generated reports)

| File | Content |
|---|---|
| `README.md` (root) | Setup (`make setup`), reproduce, serve, curl example, links |
| `00-summary.md` | One page: problem, decisions, headline results with CIs, what I would do next |
| `01-data.md` | Findings table (§0, extended), data contract, the C-feature hierarchy evidence |
| `02-features.md` | Feature catalogue: definition, train/serve source, leakage notes, ablation Δ |
| `03-text-enrichment.md` | Template finding, embedding bake-off results, Claude attribute schema, safety cross-check findings |
| `04-models-evaluation.md` | Split justification, model table, metrics with CIs, slices, calibration plots, ablations, OPE results |
| `05-ranking-policy.md` | Pipeline, gates, EV, fatigue fit, exploration and propensity math, uncertainty handling, sample rankings explained |
| `06-cold-start.md` | Signals before clicks, priors, graduation analysis and threshold |
| `07-drift-adaptation.md` | Temporal findings, staleness curve, simulator Pareto results |
| `08-serving-architecture.md` | Diagram, latency budget against measured numbers, caching/precompute/approximation, rollout |
| `09-operations.md` | Monitors, alarms, runbook (Redis down, model regression, drift alert), retraining cadence |
| `10-next-steps.md` | Data to gather, models to try, scaling |
| `adr/0001…` | Temporal split; candidate = C14 hierarchy; logloss/NE primary; DCN-v2 + LightGBM yardstick; embedding provider choice; exploration policy; pycheck rule selection; ONNX serving |
| `recording-outline.md` | Script for the video (§20) |

---

## 19. Commit sequence (atomic, conventional commits)

1. `chore: scaffold uv project, ruff, pyright, pre-commit, Makefile`
2. `ci: lint and test workflow skeleton`
3. `feat(data): pandera contracts and polars gateway with parquet cache`
4. `feat(data): temporal split and backtest folds`
5. `docs(data): EDA findings and C-feature hierarchy`
6. `feat(features): user proxy and causal counters`
7. `feat(catalog): genre and template-slot parsers`
8. `feat(features): vocabularies, context/character features, transform artifact`
9. `feat(eval): metrics, hour-block bootstrap, slices`
10. `feat(models): priors and logistic baseline`
11. `feat(models): LightGBM challenger with Optuna`
12. `feat(enrich): embedding clients, cache and bake-off`
13. `feat(enrich): Claude attribute extraction and safety cross-check`
14. `feat(models): DCN-v2 trainer, calibration, MLflow tracking`
15. `feat(eval): ablations and model comparison report`
16. `feat(eval): propensity model and SNIPS/DR`
17. `feat(ranking): gates, EV, fatigue, pacing, Thompson exploration`
18. `feat(coldstart): EB priors and graduation analysis`
19. `feat(drift): drift analysis and adaptation simulator`
20. `feat(models): ONNX export and parity test`
21. `feat(api): FastAPI rank/predict/health, Redis gateway, observability`
22. `build: Dockerfiles and compose`
23. `test(load): locust and latency report`
24. `feat(infra): Terraform modules and environments`
25. `ci: full pipeline, security, terraform, CD`
26. `docs: final docs, sample rankings, recording outline`

---

## 20. Time budget, risks, recording points

**Time (80/20):**

| Phase | Share |
|---|---|
| Data and split | 10 % |
| Features and text | 15 % |
| Models and evaluation | 30 % |
| Ranking, cold start and drift | 20 % |
| API, infrastructure and CI | 20 % |
| Docs and recording | 5 % |

The ML core gets about two-thirds of the time; infrastructure follows standard patterns.

| Risk | Mitigation |
|---|---|
| Text adds ~0 lift (likely, per §0) | Pre-registered; report it as a finding; keep the production path (embeddings at publish) because real text will not be templated |
| OPE is noisy (propensity estimated, not logged) | ESS and CIs; frame it as directional; recommend logged propensities as the first production change |
| DCN under LightGBM | Decision rule already set; ship whichever wins, both through ONNX |
| p99 under load | Measured; levers in order: fewer workers per vCPU, int8 ONNX, candidate cap, drop the per-request MC propensity (cache) |
| API-key dependence | Committed embeddings and attributes; the local Qwen path runs keyless |

**Recording narrative:**
1. The data is Avazu plus a synthetic layer.
2. Users are mostly unidentifiable.
3. The C-features hide an ad hierarchy, which defines the candidate.
4. Character signal lives in genre and tier; text is a cold-start tool, not magic.
5. Why logloss and calibration over AUC.
6. What drifts, and what that costs.
7. How exploration buys learning cheaply and safely.
8. Why the system hits <50 ms.
9. Next steps.

**Next steps (for `10-next-steps.md`):**
- Data to gather: logged propensities and full candidate sets; conversation text/sentiment at the ad moment; dwell and conversion beyond the click; real user ids with consent; creative content (image/text) embeddings; advertiser bids.
- Models to try: sequence models over the session (turn-level transformer), two-tower retrieval for candidate generation, multi-task CTR+CVR, contextual-bandit training from logged propensities, LLM-scored character–ad semantic affinity distilled into the ranker.
- Scaling: feature store (Feast on Redis/DynamoDB), streaming counters (Flink/Kinesis), GPU-free CPU serving with Triton if the model grows, regional deployments.
