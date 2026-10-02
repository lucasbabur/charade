---
title: "ADR 0001: Temporal split, test touched once"
created-at: 2026-10-01
updated-at: 2026-10-02
---

# ADR 0001: Temporal split, test touched once


## Context

CTR data is ordered in time, users and creatives repeat, and production predicts tomorrow from yesterday. A random split leaks future hours, the same user and the same creative into training.

## Decision

Train 10-21..27, validation 10-28 (early stopping, calibration, ablations), test 10-29..30, read only to confirm choices already made on validation. Hyperparameter search uses an inner split (train 21..26, select 27) so the validation day stays clean. mlcheck MLL001–003 check the split artifacts (disjoint, time-ordered, fits inside their windows); selecting on validation only is a discipline no gate can prove.

## Consequences

Validation and test are low-CTR days (16.5–17.3 %), so calibration on the most recent day matters (H9). Where a choice was first assessed on test and then redone on validation, docs/04 records it.
