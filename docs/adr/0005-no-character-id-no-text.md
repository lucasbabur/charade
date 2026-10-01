---
title: "ADR 0005: No character ID and no description text in the shipped model"
created-at: 2026-10-01
updated-at: 2026-10-01
---

# ADR 0005: No character ID and no description text in the shipped model


## Context

Ablations on validation (3-seed ensembles, paired CIs): − character ID −0.0000 [−0.0002, +0.0002]; + text (Qwen3 or TF-IDF) +0.0008 (worse); − character metadata +0.0032. The cold-start analysis finds no character-level spread beyond genre × tier.

## Decision

Drop the character-ID embedding and the conversation features; reject text features; keep genre, tier, creator type, popularity and age. Keep the embedding pipeline (Qwen3 local, OpenAI-ready) because real creator-written descriptions will not be templated.

## Consequences

New characters are scored exactly like old ones (no identity parameter to be missing). If production data shows character-level spread, the graduation procedure in docs/06 re-measures it and the ID group returns through the same ablation gate.
