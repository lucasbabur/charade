---
title: "Drift and adaptation"
created-at: 2026-10-01
updated-at: 2026-10-01
---

# 07 — Drift and adaptation

**Bottom line:** the CTR level shifts daily and ads rotate fast (13–37 % new creatives per day); a frozen model loses ~0.003 NE per day, so retrain daily. Online recalibration adds nothing. An exposure penalty selected on the validation day (λ = 8) cuts cohort concentration 38 % on the test days with no detectable CTR loss ([drift.md](../reports/drift/drift.md), [adaptation.md](../reports/drift/adaptation.md)).

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

The decision rests on the validation day (first row); the test rows were seen too, but they do not change it.

A causal per-genre logit offset updated hourly from exponentially weighted earlier residuals (half-life 6 h, shrunk toward 0):

| Day | Frozen NE | Recalibrated NE | Frozen pred/obs | Recalibrated pred/obs |
|---|---|---|---|---|
| 10-28 | 0.8712 | 0.8714 | 0.990 | 0.997 |
| 10-29 | 0.8854 | 0.8855 | 1.000 | 1.001 |
| 10-30 | 0.8859 | 0.8863 | 1.010 | 1.023 |

It nudges the level (10-28 ratio 0.990 → 0.997) and slightly worsens NE on every day, starting with the validation day that decides it. Early stopping on the most recent day already anchors the level. Not shipped. Daily retraining plus monitoring of the calibration ratio (alarm outside [0.9, 1.1]) covers the risk.

## Adaptation 2: re-balancing exposure across cohorts (prototype)

Greedy pCTR ranking concentrates each genre on the few campaigns the model likes best. The users of the dominant cohorts then see the same campaigns repeatedly: ad fatigue. The layer multiplies each candidate's score by `exp(−λ · share)`, where `share` is the campaign's share of that genre's recent allocations (exponential decay, half-life 2,000 cohort impressions). The state depends only on the policy's own choices, never on outcomes, so the replay stays a valid off-policy evaluation.

**Selection on the validation day, confirmation on test.** The rule was fixed before looking at the numbers: take the largest λ whose estimated CTR change against greedy is no worse than −0.2 pp and whose 95 % interval contains 0. DR uses the independent reward model from [05](05-ranking-policy.md).

| λ | Validation Δ vs greedy [95 % CI] | Test Δ vs greedy [95 % CI] | Test top-campaign share per genre | Test cohort HHI | Test repeat exposure |
|---|---|---|---|---|---|
| 0 (greedy) | — | — | 7.7 % | 0.027 | 10.5 % |
| 1 | +0.86 pp [+0.24, +1.47] | +0.17 pp [−0.28, +0.65] | 6.4 % | 0.023 | 10.2 % |
| 2 | +0.90 pp [+0.22, +1.50] | −0.10 pp [−0.79, +0.56] | 5.9 % | 0.021 | 10.0 % |
| 4 | +0.49 pp [−0.39, +1.39] | +0.20 pp [−0.65, +0.98] | 5.5 % | 0.019 | 9.7 % |
| **8 (selected)** | **−0.02 pp [−0.92, +0.82]** | **+0.27 pp [−0.59, +1.08]** | **4.8 %** | **0.017** | **9.4 %** |
| 16 | −0.34 pp [−1.44, +0.80] | +0.01 pp [−0.93, +0.95] | 4.1 % | 0.014 | 9.1 % |

- **λ = 8 is chosen on the validation day,** before the test days are read. On test it cuts each genre's top-campaign share from 7.7 % to 4.8 % and cohort HHI by 38 %, and repeat exposure from 10.5 % to 9.4 %. The CTR change, +0.27 pp, has an interval about ±0.8 pp wide that contains zero.
- **"Holds CTR steady" means "no loss detectable"**, not proven equivalence. The intervals are wide because the logged data supports only about 4 % effective samples.
- **Small λ looked better than greedy on the validation day** (+0.9 pp at λ = 1–2). That does not replicate on test, so it is noise or day-specific, and the rule correctly ignores it.
- **The replay cannot show the benefit it targets.** Users' exposure counters come from the logged history, not from the simulated allocation, so less repetition never raises simulated CTR. The replay can only show "no loss"; whether lower fatigue raises CTR needs an online test.
- **The earlier recommendation (λ = 4) was read off the test days.** It is replaced by this validation-day selection.
- **Path to production:**
  1. Keep the per-genre decayed campaign counts in Redis next to the user histories.
  2. Feed `exp(−λ · share)` into the candidate's value multiplier (`Candidate.pacing`, currently unused because budget pacing is not wired).
  3. A/B test λ online, because offline estimates this wide cannot settle a 0.1 pp question.
