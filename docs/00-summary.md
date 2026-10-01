# 00 — Summary

**Charade ranks ads inside AI-companion chats.** For an ad opportunity (character, chat moment, publisher, device, hour) and N candidate ads, it:
1. predicts a calibrated click probability for each candidate;
2. removes ads that must not run (brand safety, frequency caps, budget);
3. ranks the rest;
4. serves one, explores on a bounded 5 % of traffic, and logs the propensity of what it served.

It does this in **p99 26 ms at 400 rps** for 100 candidates.

## Headline results (test days 10-29/30, scored once)

| | Result | Evidence |
|---|---|---|
| CTR model | DCN-v2 3-seed ensemble, isotonic-calibrated: **NE 0.8848**, AUC 0.737, pred/obs 1.012, ECE 0.005 | [04](04-models-evaluation.md) |
| vs LightGBM / logistic | −0.0040 [−0.0052, −0.0027] / −0.0120 [−0.0138, −0.0102] log loss (paired hour-block bootstrap) | MLM004 |
| Ranking | Greedy pCTR beats the logging policy by **+1.28 pp CTR (DR, [+0.07, +2.50])**. The shipped policy with gates and exploration is +1.13 pp, CI touching zero | [05](05-ranking-policy.md) |
| Cold start | No character ID needed: characters unseen in training NE 0.906 vs 0.885 warm; new users 0.894 vs 0.856 returning | [06](06-cold-start.md) |
| Drift | Ads rotate (13–37 % new creatives per day). A frozen model loses ~0.003 NE per day, so retrain daily. The exposure penalty cuts cohort concentration 30 % with no detectable CTR loss | [07](07-drift-adaptation.md) |
| Serving | p50 8 / p95 14 / p99 26 ms at 400 rps, 0 errors; train/serve feature parity exact; ONNX = PyTorch to 1.9e-6 | [08](08-serving-architecture.md) |
| Gates | mlcheck: 42 pass, 0 fail, 4 documented warnings | [mlcheck.md](mlcheck.md) |

## What the data taught (and what it changed)

1. **Genre and safety tier are the character.**
   - Genre alone spans 14.7 % (mentor) to 23.2 % (romance) CTR, and genre × campaign interactions are real (H1).
   - Beyond genre × tier, characters are indistinguishable: the true spread is 0.0001.
   - So the shipped model has no character ID, and cold start for characters is solved by construction.
2. **Description text adds nothing here.** The descriptions are templates; the embeddings recover genre perfectly and add no CTR signal (ρ ≈ 0 ± 0.1). Adding them hurts validation log loss. The pipeline is kept for real free-text personas.
3. **Most "users" are placeholders.** 82 % of `device_id`s are one value, and 81 % of IPs appear once. The user proxy is IP + device model, and history counters exclude the current hour (the classic Avazu same-hour leak, H5).
4. **The anonymised C-columns hide the ad hierarchy:** creative → campaign → advertiser. That defines what a candidate is and where brand safety and caps attach.
5. **Conversation turn and session length are noise** (CTR flat everywhere; ablation CI includes 0). Dropped.

## How it is built

- **Code:** `src/charade/` is organised by ML concern. `tools/mlcheck` holds 46 release gates (leakage, reproducibility, recomputed model quality with CIs, parity, latency, policy invariants, drift).
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

Next steps: [10-next-steps.md](10-next-steps.md).
