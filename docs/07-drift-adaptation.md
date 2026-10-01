---
title: "Drift and adaptation"
created-at: 2026-10-01
updated-at: 2026-10-01
---

# 07 — Drift and adaptation

**Bottom line:** the CTR level shifts daily and ads rotate fast (13–37 % new creatives per day); a frozen model loses ~0.003 NE per day, so retrain daily. Online recalibration adds nothing; an exposure penalty cuts cohort concentration 30 % with no detectable CTR loss ([drift.md](../reports/drift/drift.md)).

## What shifts (daily, train → test)

| Day | CTR | First-seen-character share | Top-100 character Jaccard to the previous day | New-creative share |
|---|---|---|---|---|
| 10-22 | 16.7 % | 1.1 % | 0.75 | 34 % |
| 10-25 | 19.3 % | 0.8 % | 0.48 | 3 % |
| 10-27 | 19.6 % | 0.3 % | 0.56 | 13 % |
| 10-28 (val) | 16.5 % | 0.7 % | 0.55 | 34 % |
| 10-29 (test) | 17.3 % | 0.4 % | 0.68 | 24 % |
| 10-30 (test, partial) | 16.6 % | 0.0 % | 0.67 | 37 % |

- **The level shift is common to all genres.** On 10-28 every genre drops 2–4 pp. Romance and horror are the top two genres every day (in that order, except on the partial 10-30), and mentor is the lowest every day. The shift is in user or traffic mix, not in character preferences.
- **Character concentration is flat.** HHI is about 0.0006 every day; no character comes to dominate.
- **Feature PSI against the training window, per day after training** (`drift.json`, mlcheck MLX001 WARN):

| Feature | PSI | Cause |
|---|---|---|
| `hour_of_day` | 6.2 | Artifact: 10-30 contains only hours 00–05 |
| `C21` (advertiser) | 4.4 | Ad rotation |
| `C19` | 3.8 | Ad rotation |
| `C17` (campaign) | 2.3 | Ad rotation |
| `app_id` | 1.3 | Publisher mix |
| `C14` (creative) | 1.3 | Ad rotation |

  Context and device features stay under 0.6. Adversarial validation (train vs test, AUC 0.96) names the same culprits: creatives and campaigns.

## What it costs: staleness backtest

Each model is trained on the days before day d, early-stopped on day d, and scored on every later day (shipped configuration, 1 seed per model):

| Scored day | Age 2 | Age 3 | Age 4 | Age 5 | Age 6 | Age 7 |
|---|---|---|---|---|---|---|
| 10-29 | 0.8912 | 0.8906 | 0.8934 | 0.8978 | 0.9031 | — |
| 10-30 | 0.8879 | 0.8898 | 0.8924 | 0.8957 | 0.8978 | 0.9035 |

Age counts from the last training day (age 2 = the day after the early-stopping day). NE worsens by about 0.003 per day, and the calibration ratio wanders up to 1.10 at age 7. Decision: **retrain daily**. Training takes under a minute on one GPU, so the cost is negligible. The early-stopping and calibration day should be the most recent complete day. Promotion is gated by mlcheck ([09-operations.md](09-operations.md)).

## Adaptation 1: online recalibration (tested, not shipped)

A causal per-genre logit offset updated hourly from exponentially weighted earlier residuals (half-life 6 h, shrunk toward 0):

| Day | Frozen NE | Recalibrated NE | Frozen pred/obs | Recalibrated pred/obs |
|---|---|---|---|---|
| 10-28 | 0.8693 | 0.8695 | 1.000 | 1.000 |
| 10-29 | 0.8848 | 0.8848 | 1.010 | 1.001 |
| 10-30 | 0.8851 | 0.8854 | 1.022 | 1.026 |

It fixes the 10-29 level (ratio 1.010 → 1.001) and does nothing for NE. The calibrator fitted on the most recent day already absorbs the shift, and the within-day changes are noise at this scale. Not shipped. Daily retraining plus monitoring of the calibration ratio (alarm outside [0.9, 1.1]) covers the risk.

## Adaptation 2: re-balancing exposure across cohorts (prototype)

Greedy pCTR ranking concentrates each genre on the few campaigns the model likes best. The users of the dominant cohorts then see the same campaigns repeatedly: ad fatigue. The layer multiplies each candidate's score by `exp(−λ · share)`, where `share` is the campaign's share of that genre's recent allocations (exponential decay, half-life 2,000 cohort impressions). The state depends only on the policy's own choices, never on outcomes, so the replay stays a valid off-policy evaluation.

| λ | DR CTR | Δ vs greedy [95 % CI] | Top-campaign share per genre | Cohort HHI | Repeat exposure (same user, same campaign) |
|---|---|---|---|---|---|
| 0 (greedy) | 18.36 % | — | 7.7 % | 0.027 | 10.5 % |
| 2 | 18.32 % | −0.04 pp [−0.80, +0.68] | 5.8 % | 0.021 | 9.6 % |
| **4** | 18.26 % | **−0.10 pp [−0.98, +0.74]** | **5.4 %** | **0.019** | **9.3 %** |
| 8 | 18.12 % | −0.24 pp [−1.14, +0.61] | 4.7 % | 0.016 | 9.0 % |
| 16 | 18.15 % | −0.21 pp [−1.16, +0.71] | 4.1 % | 0.014 | 8.8 % |

The test is replayed over 102,874 impressions on the test days, on the reconstructed candidate sets of [05](05-ranking-policy.md).

- **λ = 4 is the recommended operating point.** Concentration falls 30 % and repeat exposure 11 %. The CTR estimate moves by −0.10 pp, against a CI about ±0.9 pp wide.
- **"Holds CTR steady" means "no loss detectable"**, not proven equivalence. The DR intervals are wide because the logged data supports only about 4 % effective samples.
- **Repeat exposure here is a proxy for fatigue.** The model also prices fatigue directly (`log_user_campaign_imps`), so the user-level effect is partly handled before this layer.
- **Path to production:**
  1. Keep the per-genre decayed campaign counts in Redis next to the user histories.
  2. Feed `exp(−λ · share)` into the candidate's value multiplier (`Candidate.pacing`, currently unused because budget pacing is not wired).
  3. A/B test λ online, because offline estimates this wide cannot settle a 0.1 pp question.
