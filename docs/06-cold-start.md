---
title: "Cold start"
created-at: 2026-10-01
updated-at: 2026-10-01
---

# 06 — Cold start

**Bottom line:** a character is its genre and safety tier (no measurable spread beyond them), so the shipped model has no character ID and scores new characters like old ones: test NE 0.908 cold vs 0.885 warm (n = 1,478, no CI computed; cold predictions run 6.6 % high, ECE 0.026, so the cold slice is less well calibrated than the whole) ([coldstart.md](../reports/coldstart/coldstart.md)).

## A brand-new character

| Step | What happens |
|---|---|
| Published | The character table refresh (or the request's `character` field, if the table is stale) supplies genre, safety tier, creator type, popularity and creation date |
| First request | Scored on metadata + context + device + user history. No identity parameter exists that could be missing |
| Not even metadata | OOV embeddings and the **mature** tier for brand safety (the strictest gate). `cold_start.character = true` in the response, and a metric counts it |
| Exploration | Evidence is counted per (campaign, genre), not per character, so a new character in a known genre inherits its genre's posterior width. A genuinely new genre gets the minimum evidence (20) and therefore exploration |

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

## Bootstrapping and graduation

- **Prior:** a Beta per genre × tier, fitted by the method of moments with the binomial noise removed. Its strength α + β is the number of impressions after which a character's own clicks outweigh the prior: "graduation".
- **Measured strength:** pooled over cells, the variance beyond genre × tier is below the binomial noise, so the estimate hits its numerical floor (sd 0.0001) and the implied strength, 15 million, is an artefact of that floor. It means "this estimator detects no residual character variation", not a measured number of impressions. In 19 of 30 cells the spread is likewise indistinguishable from zero; the rest give 250 to 10,000 from only 27–161 characters each, so they are noise-dominated.
- **Validation of the rule:** on validation + test, in time order, each character gets a causal logit correction learned from its own strictly earlier impressions, shrunk by τ. Using a beta-binomial strength as a logit-space penalty is a heuristic, not a derivation, so τ = 100 is shown as a sensitivity check.

| Earlier impressions of the character | Rows | Model NE | With correction, τ = pooled (15 M) | With correction, τ = 100 |
|---|---|---|---|---|
| ≤ 5 | 28,228 | 0.8805 | 0.8805 | 0.8805 |
| ≤ 20 | 44,742 | 0.8781 | 0.8781 | 0.8780 |
| ≤ 50 | 58,662 | 0.8750 | 0.8750 | 0.8750 |
| ≤ 200 | 113,203 | 0.8772 | 0.8772 | 0.8773 |
| ≤ 1,000 | 25,480 | 0.8862 | 0.8862 | 0.8868 (worse) |

**The decision this data supports:** this synthetic dataset shows no reliable gain from character-specific parameters, so the shipped model has none. A weak prior that lets characters "graduate" early makes predictions slightly worse. That is a narrower claim than "characters never graduate": with creator-written personas there may be real per-character signal. The production procedure is empirical, not the formula: refit the prior on fresh data, and give a character an ID embedding (or an online offset) only where the causal correction measurably improves NE in its evidence bucket.

## Users, by the same logic

User history passes the ablation test (+0.0009 log loss when removed, CI excludes 0). Users therefore graduate continuously, through counters rather than through an ID: every impression updates them, and `user_seen` switches the model's regime from the first repeat visit on.

## Cold ads

Characters are not the main cold-start problem here; ads are. 43 % of test rows show a creative (C14) never seen in training, and 42 % a campaign (C17) never seen in training ([coldstart.md](../reports/coldstart/coldstart.md), "Cold ads").

| Test slice | Rows | NE | Predicted / observed |
|---|---|---|---|
| All | 127,406 | 0.8855 | 1.002 |
| Creative unseen in training | 54,602 | 0.8886 | 0.946 |
| Campaign unseen in training | 53,458 | 0.8867 | 0.949 |
| Creative and campaign seen | 72,804 | 0.8891 | 1.034 |

Unseen creatives rank about as well as seen ones: they fall back to their OOV embedding plus the ad's other fields (site, app, banner position, C15–C21), which still carry signal. But calibration splits: unseen ads are under-predicted by about 5 % and seen ones over-predicted by about 3 %, which average out to 1.002 overall. The overall calibration ratio hides that, so the next model work belongs here, not in a new architecture: creative content features (text or image embeddings of the ad) and a hierarchy backoff for new campaigns. This is a standing slice of the cold-start report.
