---
title: "ADR 0006: Greedy pCTR with gates and a 5 % Thompson bucket; propensities logged"
created-at: 2026-10-01
updated-at: 2026-10-01
---

# ADR 0006: Greedy pCTR with gates and a 5 % Thompson bucket; propensities logged


## Context

The logs have no propensities, so policy quality can only be estimated offline with wide intervals. Production needs learnable, auditable data.

## Decision

Hard gates (brand safety per advertiser × tier, frequency cap, budget) → value pCTR × bid × pacing → greedy on 95 % of traffic, Thompson sampling from a (campaign, genre)-evidence Beta posterior on a hashed 5 % bucket → log the propensity of the served ad. Ties between equal calibrated pCTRs (isotonic plateaus) are broken by the raw logit.

## Consequences

DR lift over logging: greedy +1.28 pp [+0.07, +2.50]; shipped +1.13 pp [−0.09, +2.20] (gates and exploration cost ~0.15 pp). Every served decision becomes off-policy-evaluable.
