# ADR 0009: Retrain daily; do not ship online recalibration

Status: accepted (2026-10-01)

## Context

Staleness backtest: about +0.003 NE per day of model age. Online per-genre recalibration on validation + test changed NE by ≤ 0.0003.

## Decision

Retrain every day at 02:30 UTC, early-stopping and calibrating on the most recent complete day; promote only through mlcheck. Monitor the calibration ratio and alarm outside [0.9, 1.1].

## Consequences

Training costs under a minute on one GPU (about 2 minutes on CPU in the training image), so daily retraining is cheap. Recalibration code stays in the analysis as evidence, not in serving.
