---
title: "ADR 0006: Greedy pCTR with gates and 5 % exploration; exact propensities logged"
created-at: 2026-10-01
updated-at: 2026-10-01
---

# ADR 0006: Greedy pCTR with gates and 5 % exploration; exact propensities logged


## Context

The logs have no propensities, so policy quality can only be estimated offline with wide intervals. Production needs learnable, auditable data.

## Decision

Hard gates (brand safety per advertiser × tier, frequency cap) → value pCTR × bid → greedy on 95 % of traffic; on a hashed 5 % bucket, sample from q ∝ (evidence upper bound × bid)². Each candidate's probability p_i = 0.95·1[greedy] + 0.05·q_i is closed-form and logged for every candidate. Ties between equal calibrated pCTRs are broken by the raw logit, then candidate id. The budget gate and pacing multiplier exist in the library but are not wired to a spend feed.

Superseded detail: the first version used Thompson sampling and estimated the served ad's probability from the same draws that selected it, which biased propensities upward (2.6 % logged vs 1 % true in a review reproduction). Replaced because a policy whose logs cannot be trusted defeats the reason for exploring.

## Consequences

Offline DR lift over logging, under reconstructed candidate sets and frequency-share propensities: shipped +1.23 pp [+0.20, +2.20], ungated greedy +1.41 pp, with DR using an independent reward model. A recovery test and mlcheck MLP005 hold the propensities to the policy's exact distribution.
