---
title: "ADR 0003: Normalized entropy is the primary metric"
created-at: 2026-10-01
updated-at: 2026-10-01
---

# ADR 0003: Normalized entropy is the primary metric


## Context

The auction ranks by pCTR × bid. A model with better AUC but worse calibration misprices every bid.

## Decision

Primary: normalized entropy (log loss / base-rate entropy) on test. Secondary: AUC, predicted/observed overall and per day, equal-mass ECE, group AUC. Every comparison uses a paired hour-block bootstrap CI. mlcheck recomputes all of them from raw predictions (MLM003–009).

## Consequences

Some components with real AUC effects are rejected if their CI on log loss includes zero (character ID, conversation features).
