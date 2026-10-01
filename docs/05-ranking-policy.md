# 05 — Candidate ranking

**Bottom line:** for one ad opportunity with N candidates, the policy does four things. It removes candidates that must never be served (brand safety, frequency cap, exhausted budget). It ranks the rest by calibrated pCTR × bid × pacing. It serves the top one on 95 % of traffic and a Thompson sample on a hashed 5 % bucket. It logs the propensity of whatever it served. Off-policy evaluation on the test days, over reconstructed candidate sets, estimates **+1.3 pp CTR over the logging policy for greedy pCTR (DR, 95 % CI [+0.07, +2.50] pp)**, against an observed 17.2 %. For the shipped policy (gates + exploration) the estimate is +1.1 pp, with a DR CI of [−0.09, +2.21] pp that touches zero; SNIPS gives +2.1 pp [+0.90, +3.26]. Code: `charade.ranking.policy` (serving path, numpy only), `charade.analysis.policy_eval` (`uv run poe ope`). Numbers: [reports/policy/ope.md](../reports/policy/ope.md).

## Pipeline per request

| Step | Rule | Why |
|---|---|---|
| 1. Brand-safety gate | `advertiser_max_tier[C21]` ≥ character `safety_tier`, else gated | An advertiser that is family-safe must never appear next to a mature persona, whatever its pCTR. The data has no advertiser preferences, so the matrix in `[tool.charade.policy]` is illustrative (advertiser 157 ≤ suggestive, 48 = sfw only) |
| 2. Frequency cap | `prior_exposures(user, campaign) ≥ 8` → gated | CTR drops from 19.2 % on first exposure to 13–15 % after; the model already prices this (`log_user_campaign_imps`), so the cap is a user-experience limit, not a CTR lever |
| 3. Budget gate | Campaign's pacer reports its daily budget spent → gated | Never serve what cannot be billed |
| 4. Score | calibrated pCTR from one batched ONNX call | One forward pass for all N candidates |
| 5. Value | pCTR × bid × pacing multiplier | Bid defaults to 1 (pure CTR ranking, as the brief asks). Pacing (`charade.ranking.pacing.Pacer`) is a PI controller on spend against a linear daily target that throttles campaigns ahead of schedule |
| 6. Uncertainty | Beta(pCTR·n, (1−pCTR)·n), n = training impressions of (campaign, genre), clipped to [20, 1000] | A campaign never shown on romance characters gets a wide interval, so the system knows what it does not know |
| 7. Decision | Greedy on value; on the 5 % bucket `sha256(request_id)`, the first of 64 Thompson draws | Exploration spend is bounded, deterministic per request and auditable |
| 8. Propensity | (1 − ε)·1[served = greedy] + ε·P̂_TS(served), with P̂ from the same 64 draws (≥ 1/64 for an explored pick) | Every logged decision becomes usable for off-policy evaluation and unbiased retraining |

**Response per candidate:** rank (null if gated), pCTR, 90 % interval, value, gate reasons. The decision also carries `confidence: low` when the top two candidates' intervals overlap.

## Ordering when the model is uncertain

- **Exploit the posterior mean.** Greedy ranks by the mean; a lower-bound policy would lock in incumbents on 95 % of traffic.
- **Explore where intervals are wide, on a capped share.** The Thompson bucket gives uncertain candidates traffic in proportion to their chance of being best. That is where evidence is cheapest to buy (cold campaigns, cold characters).
- **Say so.** `confidence: low` lets the caller (or a business rule) prefer a safer default, and lets monitoring count how often the system is guessing.
- **Ties are deterministic:** value, then candidate id.

## Offline evaluation

The logs show one ad per impression. Candidate sets are reconstructed per publisher × hour cell: the creatives (C14 + `banner_pos`) served in that cell, top 10 by frequency. Each one has a logging propensity μ equal to its share of the cell's impressions. Every evaluated impression keeps its own user, character and context, and every candidate is re-scored for that impression, including the user's exposure count to that candidate's campaign. Coverage: 102,874 of 127,406 test impressions (80.7 %), 5,484 cells, 4.1 candidates per cell on average.

| Policy | SNIPS lift vs logging | DR lift vs logging | ESS |
|---|---|---|---|
| Uniform random | +0.37 pp [−0.00, +0.71] | −0.19 pp [−0.53, +0.13] | 17,149 |
| Greedy pCTR | +2.37 pp [+1.15, +3.62] | **+1.29 pp [+0.07, +2.50]** | 4,235 |
| **Shipped (gates + 5 % Thompson)** | +2.13 pp [+0.90, +3.26] | **+1.13 pp [−0.09, +2.21]** | 4,515 |

Paired hour-block bootstrap (1,000 resamples, the same hours for policy and logging). The observed logging CTR is 17.23 %. Greedy policies break pCTR ties (isotonic plateaus) with the raw logit, as serving does.

Reading it:
- **Random is a sanity check.** It lands where the logger does, as it should when the logger is roughly frequency-proportional.
- **The model's ranking adds about +1.3 pp (DR, about 7 % relative).** SNIPS says about +2.4 pp. DR is the more conservative estimate: it leans on the model for actions it rarely sees.
- **Gates and exploration cost about 0.16 pp** of estimated CTR against pure greedy (DR 18.36 % vs 18.52 %). That is the price of brand safety, frequency caps and the 5 % learning budget, and the reason the shipped policy's DR interval touches zero while greedy's does not.

Assumptions to keep in mind:
1. The cell share is the logging propensity, i.e. no unobserved confounding within a publisher-hour.
2. The candidate set is the served set, so a real retrieval layer offers different candidates.
3. ESS is about 4 % of rows: the estimates are directional and the intervals are honest about it.

The first production change should be logging true propensities and full candidate sets, which the serving path already does.

## Verified in code

- **Property tests** (hypothesis, 200 cases): a gated candidate is never served. The propensity is in (0, 1]. Open candidates are ordered by value. The output is identical for the same request id.
- **Exploration bucket:** hits 5 % ± 1 % over 20,000 ids.
- **mlcheck MLP003/MLP004** on 3,000 sampled decisions from the serving code path: no gated candidate served, every served ad has a propensity, and a lone eligible candidate logs propensity 1.
- **OPE estimators** recover a known policy value on synthetic logged data (IPS, SNIPS, DR CIs cover the truth).

## Sample rankings ([reports/sample_rankings.json](../reports/sample_rankings.json), `uv run poe samples`)

The same eight candidates are ranked for two chat moments through the real API:

| | Mature romance character, returning user, turn ≥ 6 | New sfw mentor character (not in the table, metadata in the request), new user, turn 1 |
|---|---|---|
| Served | `4687@1`, pCTR 0.210 [0.189, 0.231], greedy, propensity 1.0, confidence high | `15705@0` from the **exploration bucket**, propensity 0.009, confidence **low** |
| Top of the ranking | 4687@1 0.210 · 16208@1 0.149 · 20108@0 0.119 | 4687@1 0.119 · 15705@0 0.119 · 16208@1 0.117 (near-ties) |
| Gated | `19772@1` (advertiser 48, sfw-only) despite the 2nd-highest pCTR 0.162 → `brand_safety` | none: the character is sfw, so `19772@1` is eligible (0.110) |

What it shows:
- The same ads are worth much more to a romance character's returning user (0.21 vs 0.12).
- The brand-safety gate overrides pCTR.
- A brand-new character with a new user produces flat, overlapping estimates. The system says so (`confidence: low`), and this request happened to fall in the 5 % bucket, so it spent the impression learning, and logged the 0.9 % propensity that makes the outcome usable.
