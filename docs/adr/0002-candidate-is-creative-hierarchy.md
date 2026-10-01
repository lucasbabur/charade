# ADR 0002: A candidate is banner_pos + C14 and its hierarchy

Status: accepted (2026-10-01)


## Context

The brief says candidates are parameterized by `banner_pos` and "a subset of C-features". The data shows C14 determines C15, C16, C17, C21 exactly (functional dependency 1.000), while C1 and C20 vary within a creative.

## Decision

C14 = creative, C17 = campaign, C21 = advertiser, C15×C16 = size; C1 and C20 are context. A candidate is (`banner_pos`, C14, C15–C19, C21). Brand safety is per advertiser (C21), frequency caps and evidence per campaign (C17).

## Consequences

45 % of test rows show a creative unseen in training. The hierarchy gives backoff: an unseen creative usually has a known campaign or advertiser.
