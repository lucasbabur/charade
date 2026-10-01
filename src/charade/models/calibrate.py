"""Fit a calibration map on the validation day.

Candidates: identity, Platt (on the logit) and isotonic. The choice is made by 2-fold
cross-validation over validation hours (even/odd hours), so the comparison never touches test.
Calibrating on the most recent day also absorbs the day-level CTR level shift (H9).
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from charade.evaluation.metrics import logloss_rows
from charade.scoring.calibration import CalibrationKind, Calibrator


def _fit(kind: CalibrationKind, logits: np.ndarray, y: np.ndarray) -> Calibrator:
    if kind is CalibrationKind.PLATT:
        model = LogisticRegression(C=1e6).fit(logits.reshape(-1, 1), y)
        return Calibrator(kind=kind, a=float(model.coef_[0, 0]), b=float(model.intercept_[0]))
    if kind is CalibrationKind.ISOTONIC:
        p = 1 / (1 + np.exp(-logits))
        iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(p, y)
        return Calibrator(kind=kind, x=iso.X_thresholds_.tolist(), y=iso.y_thresholds_.tolist())
    return Calibrator(kind=CalibrationKind.IDENTITY)


def fit_calibrator(logits: np.ndarray, y: np.ndarray, hours: np.ndarray) -> tuple[Calibrator, dict[str, float]]:
    """Best map by held-out log loss over even/odd validation hours, refitted on all of validation."""
    folds = hours % 2 == 0
    scores: dict[str, float] = {}
    for kind in CalibrationKind:
        losses = [
            logloss_rows(y[~fold], _fit(kind, logits[fold], y[fold]).apply(logits[~fold])).sum()
            for fold in (folds, ~folds)
        ]
        scores[str(kind)] = float(sum(losses) / len(y))
    best = CalibrationKind(min(scores, key=scores.__getitem__))
    return _fit(best, logits, y), scores
