---
title: "Cold start"
created-at: 2026-10-01
updated-at: 2026-10-03
---

# 05 — Cold start

**Bottom line:** on this data, character-ID features add nothing beyond genre and safety tier in the ablation, so the shipped model has no character ID and scores new characters like old ones: test NE 0.908 cold vs 0.885 warm (n = 1,478, no CI computed; cold predictions run 6.6 % high, ECE 0.026, so the cold slice is less well calibrated than the whole) ([coldstart.md](../reports/coldstart/coldstart.md)).

## A brand-new character

| Step | What happens |
|---|---|
| Published | The character table refresh (or the request's `character` field, if the table is stale) supplies genre, safety tier, creator type, popularity and creation date |
| First request | Scored on metadata + context + device + user history. No identity parameter exists that could be missing |
| Not even metadata | OOV embeddings and the **mature** tier for brand safety (the strictest gate). `cold_start.character = true` in the response, and a metric counts it |
| Exploration | The live correction is kept per (campaign, genre), not per character, so a new character in a known genre inherits its genre's posteriors. A genuinely new genre starts every pair at the prior (multiplier 1, widest interval) and therefore explores |

## A brand-new user (or device)

There are no user-ID features by design: 82 % of rows carry a placeholder `device_id` and 81 % of IPs appear once, so an ID embedding would be mostly noise. A new user gets zero counters plus `user_seen = 0`, and the model was trained on 604k such rows (`user_seen` is one of its most used inputs). The first `POST /v1/events/impression` starts the history, and the next request already uses it (tested).

## Signal available before any click (permutation importance, NE increase when a field is shuffled)

| Characters unseen in training | Δ NE | Users with no history | Δ NE |
|---|---|---|---|
| `site_id` | +0.033 | `genre` | +0.037 |
| `app_id` | +0.027 | `app_id` | +0.037 |
| `genre` | +0.024 | `site_id` | +0.032 |
| `app_category` | +0.011 | `app_category` | +0.013 |
| `site_category`, `banner_pos` | +0.006 | `site_domain` | +0.009 |

For both kinds of cold entity, the publisher surface and the character's genre carry the signal. Ad attributes (C14, C21) matter less than where the ad runs and who the persona is.

## When would a character earn its own parameters?

- **Spread beyond genre × tier:** a beta-binomial prior per genre × tier, fitted by the method of moments with binomial noise removed. In 19 of 30 cells the spread between characters is indistinguishable from zero; the rest rest on only 27–161 characters each and are noise-dominated. Pooled, the variance beyond genre × tier is below the binomial noise.
- **Decision:** this synthetic dataset shows no reliable gain from character-specific parameters, so the shipped model has none, and the character-ID ablation agrees ([03](03-models-evaluation.md)). That is narrower than "characters never matter": creator-written personas may carry real per-character signal.
- **Rule for production:** re-run the same two checks on fresh data, the per-cell spread and the character-ID ablation on validation, and add a character embedding only if the ablation gain has a CI excluding zero. No formula for a graduation threshold is claimed; an earlier version derived one and it rested on a variance estimate that had hit its numerical floor.

## Graduation of ads

Ads are the entities that arrive cold here, so graduation is defined on them ([E012](../experiments/E012-live-correction/README.md), [04](04-ranking-policy.md#live-campaign-genre-correction-e012)). Each (campaign, genre) pair accumulates expected clicks E from the served pCTRs and observed clicks K. **Bootstrap:** while E < a, the model's pCTR dominates the pair's score and its interval is at its widest, so exploration favours it. **Graduated:** once E ≥ a = <!--n:corr_prior-->80<!--/n--> expected clicks (about 450 impressions at the base rate), the logged outcomes weigh at least as much as the prior and the observed rate decides the pair's level. The response reports `cold_start.pairs_cold` and the metric `charade_correction_pairs_total` counts both states. On the test replay from empty sums, <!--n:corr_graduated-->23 %<!--/n--> of impressions in the last hour fell on graduated pairs ([graduation.csv](../reports/coldstart/graduation.csv)); serving keys the sums by model version, so each daily bundle starts from empty too and this is the production picture. The rule was not tuned: a is the prior strength the CTR non-inferiority selection chose, and graduation falls out of it.

## Users, by the same logic

User history passes the ablation test (+0.0012 log loss when removed, CI excludes 0). Users therefore graduate continuously, through counters rather than through an ID: every impression updates them, and `user_seen` switches the model's regime from the first repeat visit on.

## Cold ads

Characters are not the main cold-start problem here; ads are. 43 % of test rows show a creative (C14) never seen in training, and 42 % a campaign (C17) never seen in training ([coldstart.md](../reports/coldstart/coldstart.md), "Cold ads").

| Test slice | Rows | NE | Predicted / observed |
|---|---|---|---|
| All | 127,406 | 0.8855 | 1.002 |
| Creative unseen in training | 54,602 | 0.8886 | 0.946 |
| Campaign unseen in training | 53,458 | 0.8867 | 0.949 |
| Creative and campaign seen | 72,804 | 0.8891 | 1.034 |

Unseen creatives rank about as well as seen ones: they fall back to their OOV embedding plus the ad's other fields (site, app, banner position, C15–C21), which still carry signal. But calibration splits: unseen ads are under-predicted by about 5 % and seen ones over-predicted by about 3 %, which average out to 1.002 overall. The overall calibration ratio hides that, so the next model work belongs here, not in a new architecture: creative content features (text or image embeddings of the ad) and a hierarchy backoff for new campaigns. This is a standing slice of the cold-start report. The live correction already closes part of the gap online: cold-campaign pred/obs <!--n:corr_cold_pred_obs-->0.937 → 0.951<!--/n--> on the test replay ([04](04-ranking-policy.md#live-campaign-genre-correction-e012)).
