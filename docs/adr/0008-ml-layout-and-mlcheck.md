---
title: "ADR 0008: Code organised by ML concern, enforced by mlcheck instead of layered architecture"
created-at: 2026-10-01
updated-at: 2026-10-02
---

# ADR 0008: Code organised by ML concern, enforced by mlcheck instead of layered architecture


## Context

Clean-architecture layers protect against swapping databases and UIs. The risks here are leakage, skew, irreproducibility and a heavy serving path.

## Decision

`src/charade/{data,features,text,models,evaluation,ranking,scoring,serving,analysis}`. Build `tools/mlcheck` (44 gates) because no maintained library covers leakage, parity, latency, propensity and holdout discipline together; use pycheck only for its layout-independent checks.

## Consequences

Every PR runs the static gates; every run (and the daily retrain) runs the full gates. Promotion happens only on a pass.
