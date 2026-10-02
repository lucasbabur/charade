---
title: "Candidate ranking"
created-at: 2026-10-01
updated-at: 2026-10-02
---

# 04 — Candidate ranking

**Bottom line:** the policy gates, ranks by pCTR × bid, serves the top ad on 95 % of traffic and explores on 5 % with a known distribution, and logs every candidate's exact selection probability. Under reconstructed candidate sets and frequency-share logging propensities, the shipped policy's estimated lift over the logging policy is **+1.46 pp CTR (DR, [+0.50, +2.38])**. This is an offline estimate under assumptions, not a measured production lift ([ope.md](../reports/policy/ope.md)).

## Pipeline per request (`charade.ranking.policy.decide`)

| Step | Rule | Why |
|---|---|---|
| 1. Brand-safety gate | `advertiser_max_tier[C21]` ≥ character `safety_tier`, else gated | An advertiser that is family-safe must never appear next to a mature persona, whatever its pCTR. The data has no advertiser preferences, so the matrix in `[tool.charade.policy]` is illustrative (advertiser 157 ≤ suggestive, 48 = sfw only) |
| 2. Frequency cap | Every exposure of the campaign to the user recorded so far, **including the current hour**, ≥ 8 → gated | A hard limit needs the live count; the model feature `log_user_campaign_imps` deliberately excludes the current hour, so the two use different counters. If Redis is unavailable there is no exposure state: the cap fails open, the response warns, and `charade_frequency_cap_unenforced_total` counts it |
| 3. Score | Calibrated pCTR from one batched ONNX call | One forward pass for all N candidates |
| 4. Value | pCTR × bid | Bid defaults to 1 (pure CTR ranking, as the brief asks) |
| 5. Evidence interval | Beta(pCTR·n, (1−pCTR)·n), n = training impressions of (campaign, genre), clipped to [20, 1000] | A heuristic width, not a calibrated posterior over prediction error. Never-seen pairings get wide intervals |
| 6. Decision | Greedy = best eligible by value, then raw logit (a step-shaped calibration map can tie pCTRs), then id. On the hashed 5 % exploration bucket, sample from q_i ∝ (upper bound_i × bid_i)² over eligible candidates | Exploration favours plausible winners and uncertain candidates, with a bounded, auditable budget |
| 7. Propensity | **Exact:** p_i = 0.95 · 1[i = greedy] + 0.05 · q_i, 0 if gated; logged for every candidate | Makes every logged decision valid for off-policy evaluation and counterfactual training |

**Response per candidate:** rank (null if gated), pCTR, interval, value, gate reasons, propensity. The decision also carries `confidence: low` when the top two intervals overlap. That is a heuristic flag for callers and monitoring, not a statistical guarantee.

**Not connected: budget pacing.** `charade.ranking.pacing.Pacer` (a PI controller on spend) and the budget gate exist and are tested, and the policy consumes `pacing` and `budget_exhausted`. But the data has no budgets or spend, so the API always passes pacing 1 and "not exhausted". Wiring it needs a spend feed and per-campaign pacer state in Redis.

## Why exact propensities

The first version explored with Thompson sampling. It estimated the served ad's probability from 64 posterior draws, including the draw that selected it, which biased the estimate upward. An external review measured it: with 100 identical candidates and full exploration, every ad's true probability is 1 %, and the mean logged value was 2.6 %. The policy now samples from an explicit distribution, so every probability is closed-form. Three checks guard it:

- **Recovery test:** logged probabilities match empirical selection frequencies within 4 standard errors over 40,000 requests (`tests/ranking/test_policy.py`). The 100-identical-candidates case logs exactly 1 %.
- **mlcheck MLP005:** in `decisions.jsonl`, per-candidate propensities sum to 1, are 0 for gated candidates, and the served ad's value equals its own entry.
- **OPE uses the serving code:** the offline evaluation calls `decide()` for every reconstructed request instead of reimplementing the policy.

## Offline evaluation (estimates under assumptions)

The logs show one ad per impression, so candidate sets are reconstructed:
- **Candidates:** for each publisher × hour cell, the top 10 creatives (C14 + `banner_pos`) served there.
- **Logging propensity μ:** each creative's share of the cell's impressions.
- **Scoring:** every evaluated impression keeps its own user, character and context, and every candidate is re-scored for it.
- **Population:** every test impression in a cell with at least two creatives: 111,368 of 127,406 (87.4 %). An impression whose logged ad is outside its cell's top 10 stays in: no evaluated policy can choose that ad, so its target probability is 0, and μ remains the ad's share of the whole cell. An earlier version dropped those rows while keeping the unconditional shares. That conditions on the logged action and biased IPS upward (eleven equal ads at 20 % CTR: IPS 22 %; reproduced in `tests/evaluation/test_ope.py`).

| Policy | SNIPS lift vs logging | DR lift vs logging | ESS |
|---|---|---|---|
| Uniform random | +0.09 pp [−0.25, +0.41] | −0.03 pp [−0.36, +0.26] | 17,149 |
| Greedy pCTR, no gates | +2.17 pp [+1.08, +3.33] | +1.65 pp [+0.72, +2.73] | 4,059 |
| Greedy pCTR with gates, no exploration | +1.97 pp [+0.85, +3.09] | +1.52 pp [+0.52, +2.50] | 4,018 |
| **Shipped policy (gates + 5 % exploration)** | +1.90 pp [+0.84, +2.97] | **+1.46 pp [+0.50, +2.38]** | 4,395 |

Paired hour-block bootstrap, 1,000 resamples; the observed logging CTR is 17.51 %. The DR direct-method term uses an **independent reward model**: a LightGBM trained on days before the last training day. Using the evaluated policy's own pCTR there would grade the model by its own beliefs (an earlier version did). On the validation day the same evaluation gives +1.96 pp [+0.68, +3.17] for the shipped policy ([ope_val.md](../reports/policy/ope_val.md)).

**What the intervals do and do not cover.** They cover sampling noise given the assumptions. They do not cover bias from the assumptions themselves:
1. Impression share within a publisher-hour is treated as the logging propensity. The target policy uses character, device and user history, so this is P(ad | cell), not P(ad | everything the policy sees): it assumes no targeting on those, and no confounding from context missing from the data. The independent reward model is a robustness check on the outcome side; it does not repair missing propensities.
2. The candidate set is assumed to be the served set.
3. ESS is about 4 % of rows, and there are only 30 test hours: the hour-block bootstrap treats the fitted models as fixed and adjacent hours as independent.
4. The frequency cap sees exposures from strictly earlier hours here (the logs have no within-hour order), while the API also counts the current hour. Same policy function, slightly different state.

Read the table as a demonstration of the evaluation machinery under these assumptions. It is not evidence of business impact and not grounds for a rollout: real candidate sets, logged propensities and outcomes (which the API now records) come first, then an online test.

- **Random lands near the logging CTR,** as expected when logging is roughly frequency-proportional.
- **What the rules cost, split** (paired DR differences, hour-block bootstrap): the gates cost 0.13 pp [−0.12, +0.37], indistinguishable from zero, and 5 % exploration costs 0.06 pp [+0.01, +0.11], small but measurable; 0.19 pp in total. The validation day agrees (0.11 and 0.08 pp).
- **Why 5 % is not optimised.** The 0.06 pp is the price of exploration in a static log; its benefit (new creatives earning enough impressions to be estimated, data the next model learns from) only appears in future traffic, which no log can show. The rate should be set from how fast new creatives need data (43 % of test impressions show one) and confirmed online; 5 % is a conventional default.
- **Read DR, not SNIPS.** DR is the headline; IPS (17.96 %) and SNIPS (19.41 %) for the shipped policy differ by more than the lift, a sign of how few heavy weights carry the importance-weighted estimates.
- **Earlier numbers:** before this fix, the shipped policy's DR interval touched zero (+1.13 pp [−0.09, +2.20]). The new exploration distribution concentrates on plausible candidates, and the evaluation now uses the serving tie-break, which moved both rows.

## Sample rankings ([reports/sample_rankings.json](../reports/sample_rankings.json), `uv run poe samples`)

| | Mature romance character, returning user, turn ≥ 6 | New sfw mentor character (not in the table, metadata in the request), new user, turn 1 |
|---|---|---|
| Served | `4687@1`, pCTR 0.210 [0.189, 0.231], greedy, propensity 0.967 | `15705@0` from the **exploration bucket**, propensity 0.008, confidence **low** |
| Top of the ranking | 4687@1 0.210 · 16208@1 0.149 · 20108@0 0.119 | 4687@1 0.119 · 15705@0 0.119 · 16208@1 0.117 (near-ties) |
| Gated | `19772@1` (advertiser 48, sfw-only), propensity 0, despite pCTR 0.162 | none: the character is sfw, so `19772@1` is eligible |

The same ads are worth much more for the romance character's returning user (0.21 vs 0.12), and the brand-safety gate overrides pCTR. For the brand-new character the estimates are flat; that request fell in the exploration bucket and logged its exact 0.8 % probability.
