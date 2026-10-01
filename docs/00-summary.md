---
title: "Summary"
created-at: 2026-10-01
updated-at: 2026-10-01
---

# 00 — Summary

**Charade ranks ads inside AI-companion chats.** For an ad opportunity (character, chat moment, publisher, device, hour) and N candidate ads, it:
1. predicts a calibrated click probability for each candidate;
2. removes ads that must not run (brand safety, frequency caps);
3. ranks the rest;
4. serves one, explores on a bounded 5 % of traffic with a known distribution, and logs every candidate's exact selection probability.

It does this in **p99 28 ms at 400 rps** for 100 candidates.

## Headline results (test days 10-29/30; Avazu sample + synthetic character layer, 30-hour test window)

| | Result | Evidence |
|---|---|---|
| CTR model | DCN-v2 3-seed ensemble, isotonic-calibrated: **NE 0.8848**, AUC 0.737, pred/obs 1.012, ECE 0.005 | [04](04-models-evaluation.md) |
| vs LightGBM / logistic | −0.0040 [−0.0052, −0.0027] / −0.0120 [−0.0138, −0.0102] log loss (paired hour-block bootstrap) | MLM004 |
| Ranking | Exact propensities logged for every candidate. Offline policy evaluation runs end to end; under reconstructed candidate sets and inferred logging propensities it estimates +1.21 pp CTR [+0.21, +2.17]. That demonstrates the evaluation machinery, not business value | [05](05-ranking-policy.md) |
| Cold start | The model has no character ID, so a new character is scored from metadata like any other. Unseen characters: NE 0.906 vs 0.885 warm (n = 1,478, known genres only); new users 0.894 vs 0.856 returning | [06](06-cold-start.md) |
| Drift | Ads rotate (13–37 % new creatives per day). A frozen model loses ~0.003 NE per day, so retrain daily. The exposure penalty cuts cohort concentration 30 % with no detectable CTR loss | [07](07-drift-adaptation.md) |
| Serving | p50 7 / p95 15 / p99 28 ms at 400 rps, 0 errors (impression and click events included); train/serve feature parity exact; ONNX = PyTorch to 1.9e-6 | [08](08-serving-architecture.md) |
| Gates | mlcheck on the shipped bundle: 47 checks, 43 pass, 0 fail, 4 warn (documented); this line is checked against `uv run mlcheck .` by `tests/docs` when a bundle is present | [mlcheck.md](mlcheck.md) |

## What the data taught (and what it changed)

1. **Genre and safety tier are the character.**
   - Genre alone spans 14.7 % (mentor) to 23.2 % (romance) CTR, and genre × campaign interactions are real (H1).
   - Beyond genre × tier, characters are indistinguishable: the true spread is 0.0001.
   - So the shipped model has no character ID: a new character has no missing parameter. That does not establish performance on new genres, shifted metadata or real conversation content.
2. **Description text adds nothing here.** The descriptions are templates; the embeddings recover genre perfectly and add no CTR signal (ρ ≈ 0 ± 0.1). Adding them hurts validation log loss. The pipeline is kept for real free-text personas.
3. **Most "users" are placeholders.** 82 % of `device_id`s are one value, and 81 % of IPs appear once. The user proxy is IP + device model, and history counters exclude the current hour (the classic Avazu same-hour leak, H5).
4. **The anonymised C-columns hide the ad hierarchy:** creative → campaign → advertiser. That defines what a candidate is and where brand safety and caps attach.
5. **Conversation turn and session length are noise** (CTR flat everywhere; ablation CI includes 0). Dropped.

## How it is built

- **Code:** `src/charade/` is organised by ML concern. `tools/mlcheck` holds 47 release gates (leakage, reproducibility, recomputed model quality with CIs, parity, latency, policy invariants, drift).
- **Settings:** all in `pyproject.toml`; tasks via `uv run poe`.
- **Decisions:** nine short ADRs in [adr/](adr/).
- **Delivery:**
  - CI on every PR: lint, strict types, about 180 tests at ≥ 85 % coverage, mlcheck, OpenAPI drift, Terraform validate/tflint/checkov, hadolint/shellcheck, image build and smoke test.
  - Every change landed through a PR into `main`, with every commit passing on its own.
  - AWS in Terraform, validated not applied; daily gated retraining ([09](09-operations.md)).

## Limits, stated plainly

- **Offline policy estimates are directional.** Propensities are inferred from impression shares, and effective samples are about 4 %. The first production change is logging real propensities and candidate sets, which the API now logs with every decision.
- **The test window is 30 hours** and the cold-character slice has 1,478 rows; those CIs are wide.
- **Not run:** OpenAI embeddings (the key had no quota), Gemini and Voyage (no keys). Given ρ ≈ 0 for two very different embedders, a third would not change the decision on this data.
- **Brand-safety preferences** are illustrative; the data has none.
- **Budget pacing is not connected to the API.** The pacer and budget gate are tested library code, but there is no spend feed to drive them.
- **The feedback loop is code, not a running job.** The API logs decisions, impressions and clicks durably, and `charade.data.events` turns them into training rows (tested end to end), but no scheduled job builds the S3 export yet.
- **Daily retraining is validated, not operated.** Rolling windows, immutable bundle promotion and redeploy are scripted and covered by Terraform validation, but they have never run on AWS.
- **`confidence` and the evidence intervals are heuristics** derived from training support, not calibrated posteriors.
- **External reviews found three correctness bugs, all fixed with tests:** Thompson-sampling propensities biased upward; counters that let later-hour events leak into earlier snapshots; and a frequency cap that ignored the current hour, because it reused the causal model feature. The cap now counts every recorded exposure; the feature still excludes the current hour.
- **MLL006 is a budget, not a proof.** It caps how many distinct configurations look at one holdout window; it cannot prove results were not used to choose among them.
- **Feedback events are at-least-once.** The API re-emits an event when a retried write finds it already stored, and the row builder deduplicates; a transactional outbox (append to a Redis stream inside the same MULTI, delivered by a separate consumer) is the production version.
- **Synthetic layer.** The genre-driven findings come from a synthetic character layer on Avazu; they will not transfer as-is to real companion conversations. The ablation and graduation procedures are what transfer.

Next steps: [10-next-steps.md](10-next-steps.md).
