# ADR 0004: DCN-v2 seed ensemble ships; LightGBM and logistic are yardsticks

Status: accepted (2026-10-01)

## Context

H1: genre × campaign interactions are real (residual sd 2.7 pp vs 1.1 pp noise). The ranking question is about crosses.

## Decision

Ship a 3-seed DCN-v2 ensemble (tuned by Optuna on the inner split), isotonic-calibrated, exported to one ONNX graph. Always train a logistic baseline and a tuned LightGBM on the same features and report them.

## Consequences

Test NE 0.8848 vs LightGBM 0.8934 (−0.0040 [−0.0052, −0.0027]) and logistic 0.9110. Serving needs no torch: ONNX Runtime scores 100 candidates in ~1.6 ms.
