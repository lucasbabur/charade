# 06 — Cold start

**Bottom line:** a character is its genre and safety tier (no measurable spread beyond them), so the shipped model has no character ID and scores new characters like old ones: test NE 0.906 cold vs 0.885 warm (n = 1,478, no CI computed) ([coldstart.md](../reports/coldstart/coldstart.md)).

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
- **Measured strength:** pooled over cells it is **15 million** (true sd 0.0001). In 19 of 30 cells the true spread is indistinguishable from zero (strength above 10^5). The remaining cells give 250 to 10,000, but each rests on only 27–161 characters, so they are noise-dominated.
- **Validation of the rule:** on validation + test, in time order, each character gets a causal logit correction learned from its own strictly earlier impressions, shrunk by τ.

| Earlier impressions of the character | Rows | Model NE | With correction, τ = pooled (15 M) | With correction, τ = 100 |
|---|---|---|---|---|
| ≤ 5 | 28,228 | 0.8794 | 0.8794 | 0.8794 |
| ≤ 50 | 58,662 | 0.8734 | 0.8734 | 0.8735 |
| ≤ 200 | 113,203 | 0.8760 | 0.8760 | 0.8761 |
| ≤ 1,000 | 25,480 | 0.8847 | 0.8847 | 0.8853 (worse) |

**The decision this data supports:** characters never graduate, because there is nothing character-specific to learn. A weak prior that lets characters "graduate" early makes predictions slightly worse. In production, where creator-written personas will differ, the same procedure sets the threshold: refit the pooled prior on fresh data. A character graduates (gets an ID embedding at the next retrain, or an online offset) once its impressions exceed α + β, if and only if the causal correction improves NE in the bucket it is entering. The rule is evidence-based and re-measured, not a constant.

## Users, by the same logic

User history passes the ablation test (+0.0009 log loss when removed, CI excludes 0). Users therefore graduate continuously, through counters rather than through an ID: every impression updates them, and `user_seen` switches the model's regime from the first repeat visit on.
