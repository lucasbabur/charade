---
title: "Drift and adaptation"
created-at: 2026-10-01
updated-at: 2026-10-03
---

# 06 — Drift and adaptation

**The rolling-window backtest is descriptive, not an isolated effect of model age.** Training volume and traffic mix vary; NE, AUC and calibration point estimates do not establish ranking stability or an optimal retraining cadence. Daily retraining remains an operational choice for new creatives and calibration monitoring. The validation-selected exposure penalty (λ = 2) cuts cohort concentration 24 % on test, but does not confirm CTR non-inferiority ([drift.md](../reports/drift/drift.md), [adaptation.md](../reports/drift/adaptation.md)). The shipped adaptation layer is the live (campaign, genre) correction: it follows outcomes between retrains and is non-inferior on test (<!--n:corr_dr-->+0.49 pp [−0.07, +1.02]<!--/n--> DR vs greedy).

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

Each model is trained on the **3 days** before day d, early-stopped on day d, and scored on every later day (shipped configuration, a 2-seed mean-logit ensemble). Duration is fixed; training counts, validation counts and traffic mix are not. The generated report includes these counts, scored rows, NE, across-impression AUC and calibration. This is a rolling-window comparison, not a controlled age-only experiment:

| Scored day | Age 2 | Age 3 | Age 4 | Age 5 | Age 6 | Age 7 |
|---|---|---|---|---|---|---|
| 10-29 NE | 0.8993 | 0.8968 | 0.8972 | 0.8955 | 0.9009 | — |
| 10-30 NE | 0.8936 | 0.9130 | 0.9019 | 0.9069 | 0.8925 | 0.9011 |
| 10-30 pred/obs | 0.969 | 1.069 | 1.054 | 1.032 | 1.056 | 1.115 |

Age counts from the last training day (age 2 = the day after the early-stopping day). The within-scored-day slopes are descriptive point estimates: NE +0.00002 and pred/obs +0.012 per day. No paired uncertainty or equivalence margin was computed for these trends. NE combines discrimination and calibration; even across-impression AUC cannot establish within-opportunity ranking stability. A flat NE slope is not evidence that ranking is unchanged.

**Correction.** The earlier expanding-window experiment confounded model age with the number of training days. Fixed-duration windows reduce that problem but retain unequal row counts and changing traffic mix. Neither experiment identifies how much of the old slope came from data volume versus staleness; the earlier causal attribution is withdrawn. An age-only comparison needs matched training counts and paired uncertainty on the same scored rows.

Decision: **retrain daily** as an operational default to refresh new creatives and monitor calibration, not as a cadence proven optimal by this experiment. Validate the cadence with actual label-availability and deployment delays before production. Promotion remains gated by mlcheck ([07](07-serving-operations.md)).

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

**Selection on the validation day, confirmation on test.** The rule was fixed before looking at the numbers: take the largest λ whose 95 % lower bound on the CTR change against greedy clears −0.2 pp (non-inferiority with a 0.2 pp margin). An earlier rule accepted any λ whose interval merely contained 0, which bounds no loss at all. DR uses the independent reward model from [04](04-ranking-policy.md).

| λ | Validation Δ vs greedy [95 % CI] | Test Δ vs greedy [95 % CI] | Test top-campaign share per genre | Test cohort HHI | Test repeat exposure |
|---|---|---|---|---|---|
| 0 (greedy) | — | — | 8.3 % | 0.029 | 10.3 % |
| 1 | +0.85 pp [+0.32, +1.36] | +0.10 pp [−0.33, +0.56] | 6.9 % | 0.024 | 10.0 % |
| **2 (selected)** | **+0.81 pp [+0.28, +1.28]** | **+0.06 pp [−0.60, +0.74]** | **6.3 %** | **0.022** | **9.7 %** |
| 4 | +0.37 pp [−0.35, +1.00] | −0.12 pp [−0.89, +0.70] | 5.8 % | 0.020 | 9.4 % |
| 8 | +0.52 pp [−0.30, +1.34] | −0.23 pp [−0.85, +0.40] | 5.1 % | 0.017 | 9.1 % |
| 16 | −0.34 pp [−1.37, +0.68] | −0.29 pp [−1.23, +0.81] | 4.4 % | 0.015 | 8.9 % |

- **λ = 2 is chosen on the validation day,** before the test days are read: it is the largest λ whose lower bound (+0.28 pp) clears the margin. Under the old rule the chosen λ was 8; that changed because the OPE population fix (see [04](04-ranking-policy.md)) moved every row and the rule became a real non-inferiority test.
- **Test does not confirm non-inferiority.** On test, λ = 2 cuts each genre's top-campaign share from 8.3 % to 6.3 %, cohort HHI by 24 % and repeat exposure from 10.3 % to 9.7 %, at +0.06 pp CTR. But the lower bound, −0.60 pp, does not clear −0.2 pp: the data cannot rule out a loss of that size. Status: candidate for an online test, not a shipped change.
- **The validation-day gain does not replicate** (+0.8 pp at λ = 1–2 on validation, ≈ 0 on test): noise or day-specific.
- **The replay cannot show the benefit it targets.** Users' exposure counters come from the logged history, not from the simulated allocation, so less repetition never raises simulated CTR. The replay can only show "no loss"; whether lower fatigue raises CTR needs an online test.
- **The earlier recommendation (λ = 4) was read off the test days.** It is replaced by this validation-day selection.
- **Path to production:**
  1. Keep the per-genre decayed campaign counts in Redis next to the user histories.
  2. Feed `exp(−λ · share)` into the candidate's value multiplier (`Candidate.pacing`, currently unused because budget pacing is not wired).
  3. A/B test λ online, because offline estimates this wide cannot settle a 0.1 pp question.

## Adaptation 3: live (campaign, genre) correction (shipped)

Between daily retrains the model's pCTR is frozen while campaigns rotate and fatigue. The correction of [E012](../experiments/E012-live-correction/README.md) multiplies each candidate's pCTR by a per-(campaign, genre) posterior mean learned from the served pCTRs and observed clicks, updated inside the impression and click transactions. Replayed on the test days from empty sums, prior <!--n:corr_prior-->80<!--/n--> selected on validation: DR CTR change vs greedy <!--n:corr_dr-->+0.49 pp [−0.07, +1.02]<!--/n-->, logged-ad log loss <!--n:corr_logloss-->−0.00054 [−0.00087, −0.00022]<!--/n-->, cohort HHI <!--n:corr_hhi-->0.0275<!--/n--> against 0.0293 for greedy, without targeting concentration. A half-life on the sums exists for week-scale drift but cannot be distinguished from none on a 30-hour window, so the shipped setting keeps every outcome. Unlike the exposure penalty, this layer reacts to outcomes, so its replay is a lower bound on what it learns in production, where it also sees the outcomes of its own choices. Details and the two readings: [04](04-ranking-policy.md#live-campaign-genre-correction-e012).
