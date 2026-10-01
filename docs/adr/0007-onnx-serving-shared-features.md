# ADR 0007: ONNX serving and one shared feature transform

Status: accepted (2026-10-01)


## Context

Train/serve skew and heavy serving images are the two classic failure modes.

## Decision

Serving imports `charade.features` (derive + encode) and `charade.scoring` (ONNX Runtime, numpy calibration); it never imports torch, lightgbm or sklearn. User counters have one definition, implemented offline in polars and online in `UserHistory`.

## Consequences

Parity is exact on 2,000 replayed requests (MLV001), ONNX matches torch to 1.9e-6 (MLV002), p99 26 ms at 400 rps. mlcheck MLS001/MLS002 enforce the import rules on every PR.
