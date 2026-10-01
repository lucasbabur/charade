---
title: "ADR 0004: DCN-v2 seed ensemble ships; LightGBM and logistic are yardsticks"
created-at: 2026-10-01
updated-at: 2026-10-01
---

# ADR 0004: DCN-v2 seed ensemble ships; LightGBM and logistic are yardsticks


## Context

H1: genre × campaign interactions are real (residual sd 2.7 pp vs 1.1 pp noise). The ranking question is about crosses.

## Decision

Ship a 3-seed DCN-v2 ensemble (tuned by Optuna on the inner split), exported to one ONNX graph, with the simplest calibration map that is not beaten by more than noise (identity). Always train a logistic baseline and a LightGBM ensemble with the same tuning effort (60 trials) and the same seeds, and report ensemble-vs-ensemble and single-vs-single comparisons.

## Consequences

Test NE 0.8855 vs LightGBM ensemble 0.8942 (−0.0040 [−0.0053, −0.0027]) and logistic 0.9110. Single model against single model the gap is −0.0020 [−0.0041, +0.0004]: the ensemble result is established, an architecture advantage over boosted trees is not. Serving needs no torch: ONNX Runtime scores 100 candidates in ~1.6 ms.
