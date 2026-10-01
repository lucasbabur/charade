# 01 — Data

**Bottom line:** a 1M-row Avazu sample (2014-10-21..30) plus a synthetic character layer that carries real signal (genre, tier) and a conversation layer that is noise. Most users are unidentifiable, and the C-columns hide the ad hierarchy that defines a candidate ([eda.md](../reports/eda/eda.md)).

## Loading contract (`charade.data.load`)

| Rule | Why |
|---|---|
| Read columns **by name** (`schema_overrides`), reject missing or extra columns | Reading by position silently shifted every column once during development (C1 landed in `banner_pos`); a test now reorders the CSV columns and expects identical output |
| All ids stay strings | Hashes with leading zeros; `id` beyond int64 semantics |
| Unique `id`, `click` ∈ {0,1}, `1 ≤ turn ≤ session_msg_count` | Duplicate ids double-count clicks; the other two are invariants features assume |
| Every `character_id` resolves; impression ≥ `created_at` | A broken join must fail, not quietly become "cold start" |
| Output sorted by (`ts`, `id`) | Deterministic order for counters and replays |

`uv run poe mlcheck-data` re-checks the contract from outside the package (MLD001–010).

## Findings and the decision each drives

| Finding (train split) | Decision |
|---|---|
| Daily CTR 16.7–19.6 %; validation day 16.5 %, test days 17.3 % / 16.6 % | Calibrate on the most recent day (val); report the per-day calibration ratio (MLM006) |
| 2014-10-30 holds 23k rows (partial day) | Kept in test; never read alone |
| `device_id = a99f214a` on 82.5 % of rows | Placeholder, not a user. User proxy = real `device_id`, else `device_ip \| device_model`. The `device_id_real` feature maps the placeholder to its own token |
| Genre (name prefix): mentor 14.7 %, romance 23.2 %, horror 22.5 %, others 17.7–18.2 % | Genre is a first-class categorical; strongest pre-click character signal |
| Safety tier: mature 20.5 % vs sfw 18.1 %, suggestive 18.3 % | Feature and brand-safety gating dimension |
| Genre × campaign residual sd 2.7 pp vs binomial SE 1.1 pp; tier × campaign 1.6 pp vs 0.7 pp | Interactions are real → a cross network (DCN-v2), and the ranking must be per character |
| C14 → C15, C16, C17, C21 deterministic; C17 → C21 deterministic; C1, C20 vary within a creative | C14 = creative, C17 = campaign, C21 = advertiser. **Candidate = `banner_pos` + C14 and its derived fields.** C1 and C20 are context |
| Conversation turn: 18.4–18.7 % in every bucket | Behind the `conversation` group; ablated (H7) |
| Users seen before: 15–17 % vs 19.1 % first-time | User-history counters (H4) |
| Same-hour user impression count: 19.3 % at 1 → 9.8–14 % at ≥ 2 | **Leak**: the hour's total is only known after the hour. Counters use strictly earlier hours (H5) |
| Repeat exposure to a campaign: 19.5 % first → 13–15 % after | Fatigue feature and ranking penalty (H6) |
| Test OOV: C14 45 %, C17 42 %, C21 12 % of test rows; characters 3 % | Ad rotation is fast. The hierarchy gives backoff: a new creative still has a known size, campaign or advertiser most of the time; tracked in drift |

## Split

Train = 2014-10-21 → 10-27 (729,685 rows, every weekday once), validation = 10-28 (142,909), test = 10-29 → 10-30 05h (127,406). The split is temporal; random or k-fold splits leak future hours, shared users and repeated creatives into training (mlcheck MLS003, MLL001–002). Boundaries live in `[tool.charade]`.

## Test fixture

`tests/fixtures/` holds every impression of ~2,500 randomly sampled users plus the 25 heaviest (16.7k rows), with their characters. Sampling **users, not rows** keeps histories intact so counter parity is testable. Rebuild with `uv run poe fixture`.
