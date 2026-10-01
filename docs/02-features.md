# 02 — Features

**Bottom line:** the shipped model uses 25 categorical fields (embedded, out-of-vocabulary index 0, vocabularies fitted on training rows only) and 10 dense inputs (standardized on training). There is one definition, in `charade.features`, called by training, the API, parity and every analysis. Feature groups were kept or dropped by validation ablations ([04](04-models-evaluation.md)). The single-feature AUCs below come from `leakage.json`; none exceeds 0.67 (MLL005).

## Shipped feature set

| Group (ablation Δ log loss if removed) | Fields | Definition and source | Leakage / serving notes |
|---|---|---|---|
| Context (not ablated: core) | `hour_of_day`, `site_id`, `site_domain`, `site_category`, `app_id`, `app_domain`, `app_category`, `C1`, `C20` | Raw request fields; hour from the request timestamp | Strongest single features: `site_id` AUC 0.671, `site_domain` 0.660. C1 and C20 vary within a creative, so they belong to context ([ADR 0002](adr/0002-candidate-is-creative-hierarchy.md)) |
| Device (+0.0019 [+0.0015, +0.0026]) | `device_model`, `device_type`, `device_conn_type`, `device_id_real` | `device_id_real` maps the placeholder `a99f214a` to its own token; real ids need ≥ 20 training rows | Placeholder on 82 % of rows ([01](01-data.md)) |
| Ad (candidate side; not ablated: core) | `banner_pos`, `C14`–`C19`, `C21` | From each candidate in the request | 45 % of test creatives are new to training: OOV for C14, backoff through C17 and C21 |
| Character metadata (+0.0032 [+0.0027, +0.0037]) | `genre`, `safety_tier`, `creator_type`, `interactions_bucket` (log2 of `num_interactions`), dense `log_num_interactions`, `log_character_age_days` | Character table; genre = name prefix; age = impression time − `created_at` | `num_interactions` is a snapshot (possible future information); its standalone AUC is 0.503, so it carries no leak worth worrying about |
| User history (+0.0009 [+0.0004, +0.0013]) | `user_seen`, `log_user_imps`, `log_user_clicks`, `user_ctr_logit` (Beta(0.72, 3.28)-smoothed), `log_user_imps_24h`, `log_user_clicks_24h`, `log_hours_since_last`, `log_user_campaign_imps` (per candidate) | User = real device id, else `device_ip \| device_model`. **Strictly earlier hours only** | Same-hour counts are the known Avazu leak (H5) and are excluded. Offline (polars) and online (`UserHistory`, Redis) definitions are parity-tested, including hypothesis-generated event orders |

## Built and rejected

| Group | Fields | Ablation | Why rejected |
|---|---|---|---|
| Character ID | `character_id` embedding (min count 20, 8 % ID dropout) | −0.0000 [−0.0002, +0.0002] | No character-level signal beyond genre × tier ([06](06-cold-start.md)) |
| Conversation | `turn_bucket`, `session_bucket`, `log_turn`, `turn_position` | +0.0001 [−0.0000, +0.0003] | CTR is flat across turns. `session_msg_count` ("total messages in session") may also not be known mid-session |
| Text | 16-dim PCA of Qwen3 or TF-IDF description embeddings | +0.0008 [+0.0006, +0.0011] (worse) | Templated descriptions ([03](03-text-enrichment.md)) |

## Encoding rules

- Vocabularies: values with at least 5 training rows (20 for ids). Everything else, including every unseen value, goes to index 0.
- Dense: nulls and non-finite values become 0 before standardization. That is a defensive guard, after the load test found an out-of-order-event bug that produced −∞.
- Categorical encoding for ≤ 2,000 rows uses dictionary lookups (the serving path), larger frames use polars `replace_strict`. Both produce identical ids (MLV001).
- `feature_spec.json` (in the bundle) stores the fields, vocabularies, scaling and groups. The API reads it; nothing is re-fitted at serve time.
