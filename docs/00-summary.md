---
title: "Summary"
created-at: 2026-10-01
updated-at: 2026-10-03
---

# 00 — Summary

**Charade ranks ads inside AI-companion chats.** For an ad opportunity (character, chat moment, publisher, device, hour) and N candidate ads, it:
1. predicts a calibrated click probability for each candidate;
2. removes ads that must not run (brand safety, frequency caps);
3. ranks the rest;
4. serves one, explores on a bounded 5 % of traffic with a known distribution, and logs every candidate's exact selection probability.

It does this in **p99 25–31 ms at 400 rps** for 100 candidates, measured on one desktop.

## Headline results (test days 10-29/30; Avazu sample + synthetic character layer, 30-hour test window)

| | Result | Evidence |
|---|---|---|
| CTR model | DCN-v2 3-seed ensemble: **NE 0.8855**, AUC 0.737, pred/obs 1.002, ECE 0.005 (no calibration map: none beat identity by more than noise) | [03](03-models-evaluation.md) |
| vs LightGBM / logistic | Equal tuning effort, 3-seed ensembles each: −0.0040 [−0.0053, −0.0027] vs LightGBM, −0.0117 [−0.0136, −0.0098] vs logistic. Single model vs single model: −0.0020 [−0.0041, +0.0004], not established | MLM004 |
| Ranking | Exact propensities logged for every candidate. Offline policy evaluation runs end to end; under reconstructed candidate sets and inferred logging propensities it estimates +1.46 pp CTR [+0.50, +2.38] (DR with an independent reward model). That demonstrates the evaluation machinery, not business value | [04](04-ranking-policy.md) |
| Cold start | The model has no character ID, so a new character is scored from metadata like any other. Unseen characters: NE 0.908 vs 0.885 warm (n = 1,478, known genres only); new users 0.894 vs 0.856 returning | [05](05-cold-start.md) |
| Drift | Ads rotate (13–37 % new creatives per day); a frozen model loses ~0.003 NE per day, so retrain daily. An exposure penalty chosen on the validation day by a non-inferiority rule (λ = 2) cuts cohort concentration 24 % on test, but test cannot rule out a 0.6 pp CTR loss, so it awaits an online test | [06](06-drift-adaptation.md) |
| Serving | p50 8 / p95 14–15 / p99 25–31 ms at 400 rps over two runs, errors ≤ 0.005 % (impression events over the 10 ms store budget; impression and click events included); train/serve feature parity exact; ONNX = PyTorch to 1.9e-6 | [07](07-serving-operations.md) |
| Gates | mlcheck on the shipped bundle: 44 checks, 40 pass, 0 fail, 4 warn (documented); this line is checked against `uv run mlcheck .` by `tests/docs` when a bundle is present | [mlcheck.md](mlcheck.md) |

## What the data taught (and what it changed)

1. **Genre and safety tier carry the character signal.**
   - Genre alone spans 14.7 % (mentor) to 23.2 % (romance) CTR, and genre × campaign interactions are real (H1).
   - Character-ID features did not improve validation log loss in the ablation (−0.0001 [−0.0006, +0.0002], both seed sets), so the shipped model uses metadata instead. That is a finding about this data, not a claim that characters are interchangeable.
   - So the shipped model has no character ID: a new character has no missing parameter. That does not establish performance on new genres, shifted metadata or real conversation content.
2. **Description text adds nothing here.** The descriptions are templates; the embeddings recover genre perfectly and add no CTR signal (ρ ≈ 0 ± 0.1). Adding them hurts validation log loss. The pipeline is kept for real free-text personas.
3. **Most "users" are placeholders.** 82 % of `device_id`s are one value, and 81 % of IPs appear once. The user proxy is IP + device model, and history counters exclude the current hour (the classic Avazu same-hour leak, H5).
4. **The anonymised C-columns hide the ad hierarchy:** creative → campaign → advertiser. That defines what a candidate is and where brand safety and caps attach.
5. **Conversation turn and session length are noise** (CTR flat everywhere; ablation CI includes 0). Dropped.

## Decisions

One line each; the reasoning and the numbers live in the linked page.

| Decision | Why | Evidence |
|---|---|---|
| Split by time: train 10-21..27, validate 10-28, test 10-29/30 read only to confirm; tune on an inner split (select on 10-27) | Production predicts the future; random splits leak hours, users and creatives | [01](01-data.md), [03](03-models-evaluation.md) |
| A candidate is a creative (`banner_pos`, C14) with its campaign (C17) and advertiser (C21) | The C-columns are a deterministic hierarchy; brand safety attaches to the advertiser, caps and evidence to the campaign | [01](01-data.md) |
| Normalized entropy is the primary metric, with AUC, pred/obs and ECE; every comparison gets a paired hour-block bootstrap CI | Ads are priced on calibrated probabilities, not just ordering; rows within an hour are correlated | [03](03-models-evaluation.md) |
| Ship a 3-seed DCN-v2 ensemble in one ONNX graph, uncalibrated (identity); LightGBM and logistic trained with equal effort as yardsticks | Best ensemble result, explicit feature crosses, cheap to serve; the architecture-alone advantage is not established (E004, E010) | [03](03-models-evaluation.md) |
| No character ID, no conversation features, no description text | Each was ablated with two seed sets and did not help (text hurt) | [02](02-features.md), [03](03-models-evaluation.md) |
| Gate (brand safety, frequency cap), rank by pCTR × bid, serve greedy on 95 %, explore 5 % from a known distribution, log every candidate's exact propensity | Hard business rules never lose to a score; exact propensities make every decision usable for off-policy evaluation | [04](04-ranking-policy.md) |
| One feature implementation shared by training and serving; serving never imports torch, LightGBM or sklearn | Train/serve skew is the classic failure; a light serving image keeps p99 low | [02](02-features.md), [07](07-serving-operations.md) |
| Organise code by ML concern and gate releases with mlcheck, not with clean-architecture layers | The risks here are leakage, skew and irreproducibility, not swapping databases | [mlcheck.md](mlcheck.md) |
| Retrain daily on a rolling window; no online recalibration | A frozen model loses ~0.003 NE per day; recalibration did not help on the validation day | [06](06-drift-adaptation.md) |

## How it is built

- **Code:** `src/charade/` is organised by ML concern. `tools/mlcheck` holds 44 release gates (leakage, reproducibility, recomputed model quality with CIs, parity, latency, policy invariants, drift).
- **Settings:** all in `pyproject.toml`; tasks via `uv run poe`.
- **Delivery:**
  - CI on every PR: lint, strict types, import layers, dead code, 222 tests at ≥ 85 % coverage plus 110 for mlcheck, mlcheck, OpenAPI drift, Terraform validate/tflint/checkov, hadolint/shellcheck, image build and smoke test.
  - Every change landed through a PR into `main`, with every commit passing on its own.
  - A Terraform sketch of the AWS serving environment, validated not applied; retraining and promotion as scripts ([07](07-serving-operations.md)).

## Limits, stated plainly

- **Offline policy estimates are directional.** Propensities are inferred from impression shares, and effective samples are about 4 %. The first production change is logging real propensities and candidate sets, which the API now logs with every decision.
- **The test window is 30 hours** and the cold-character slice has 1,478 rows; those CIs are wide.
- **Not run:** paid embedding APIs (OpenAI, Gemini, Voyage: no usable keys). Given ρ ≈ 0 for two very different embedders, a third would not change the decision on this data.
- **Brand-safety preferences** are illustrative; the data has none.
- **Budget pacing is not connected to the API.** The pacer and budget gate are tested library code, but there is no spend feed to drive them.
- **The feedback loop is code, not a running job.** The API logs decisions, impressions and clicks durably, and `charade.data.events` turns them into training rows (tested end to end), but no scheduled job builds the S3 export yet.
- **Infrastructure is a sketch, not a platform.** Rolling windows, gated promotion and pinned-bundle rollback are scripted; the serving environment passes Terraform validation, tflint and checkov. None of it has run on AWS, and the scheduler, log export and CI deploy credentials were deliberately left out.
- **`confidence` and the evidence intervals are heuristics** derived from training support, not calibrated posteriors.
- **External reviews found three correctness bugs, all fixed with tests:** Thompson-sampling propensities biased upward; counters that let later-hour events leak into earlier snapshots; and a frequency cap that ignored the current hour, because it reused the causal model feature. The cap now counts every recorded exposure; the feature still excludes the current hour.
- **Historical click features assume clicks arrive before the next hour.** The 48 h label window keeps immature labels out of training rows, but the user counters still count an earlier impression's click as known at the next request, which live serving cannot guarantee. Measuring the gap needs real click delays.
- **The bundle digest (MLR005) binds evidence to model files, not to serving code or policy settings.** A code or `[tool.charade.policy]` change after the evidence was produced is caught by code review and the commit in the manifest, not by that gate.
- **Holdout discipline is not enforced by a gate.** Choices are made on validation and test only confirms; where that was violated and redone, docs/03 says so. An earlier ledger-based budget gate was removed because it did not track every analysis that read test.
- **Feedback events are at-least-once.** The API re-emits an event when a retried write finds it already stored, and the row builder deduplicates; a transactional outbox (append to a Redis stream inside the same MULTI, delivered by a separate consumer) is the production version.
- **Synthetic layer.** The genre-driven findings come from a synthetic character layer on Avazu; they will not transfer as-is to real companion conversations. The ablation procedure is what transfers.
- **The full Kaggle Avazu file was used once, after every decision** (E010), to check whether the model family holds up with 9x the data. Nothing shipped was trained, tuned or selected on it; it covers the same ten days, so training on it would leak the test window.
- **This is not Simula's core problem.** Avazu ads are anonymous ids and the characters are templates, so nothing here can test *semantic* matching: a toy-rocket ad next to a space-ranger character, or an ad that fits what the user is talking about right now. The model learns id-level affinities (genre × campaign) instead. Charade demonstrates the pipeline around such a model (split discipline, calibrated scoring, gated ranking with exact propensities, feedback, serving); the matching itself needs ad content and conversation embeddings ([08](08-next-steps.md)).

Next steps: [08-next-steps.md](08-next-steps.md).
