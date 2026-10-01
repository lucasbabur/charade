---
title: "ADR 0001: Temporal split, test touched once"
created-at: 2026-10-01
updated-at: 2026-10-01
---

# ADR 0001: Temporal split, test touched once


## Context

CTR data is ordered in time, users and creatives repeat, and production predicts tomorrow from yesterday. A random split leaks future hours, the same user and the same creative into training.

## Decision

Train 10-21..27, validation 10-28 (early stopping, calibration, ablations), test 10-29..30 under a configuration budget (each distinct configuration scored on it is logged and counted). Hyperparameter search uses an inner split (train 21..26, select 27) so the validation day stays clean. Enforced by mlcheck MLS003, MLL001–003 and MLL006 (a budget, which limits but cannot prove the absence of test-driven selection).

## Consequences

Validation and test are low-CTR days (16.5–17.3 %), so calibration on the most recent day matters (H9). Every test scoring is logged with its configuration hash in `evaluation_ledger.jsonl`; reruns of an identical configuration are free, new configurations count against the budget. The first exploratory run also scored test before the ablations; that is recorded in docs/04.
