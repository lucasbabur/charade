"""Metrics. Primary: normalized entropy (log loss / base-rate entropy); the auction consumes pCTR x bid."""

import math

import numpy as np
import numpy.typing as npt
from sklearn.metrics import roc_auc_score

type Floats = npt.NDArray[np.float64]
_EPS = 1e-12


def logloss_rows(y: Floats, p: Floats) -> Floats:
    """Per-row binary cross-entropy."""
    p = np.clip(p, _EPS, 1 - _EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def entropy(rate: float) -> float:
    """Bernoulli entropy in nats."""
    return 0.0 if rate in {0.0, 1.0} else -(rate * math.log(rate) + (1 - rate) * math.log(1 - rate))


def normalized_entropy(y: Floats, p: Floats) -> float:
    """Log loss over the entropy of this set's base rate; 1.0 = constant base-rate predictor."""
    return float(logloss_rows(y, p).mean()) / entropy(float(y.mean()))


def ece(y: Floats, p: Floats, bins: int = 15) -> float:
    """Expected calibration error with equal-mass bins."""
    order = np.argsort(p, kind="stable")
    return float(sum(len(c) / len(p) * abs(p[c].mean() - y[c].mean()) for c in np.array_split(order, bins) if len(c)))


def group_auc(y: Floats, p: Floats, groups: npt.NDArray[np.int64]) -> float:
    """Impression-weighted AUC within groups that contain both classes (ranking quality per user-hour)."""
    order = np.argsort(groups, kind="stable")
    y, p, groups = y[order], p[order], groups[order]
    bounds = np.flatnonzero(np.diff(groups)) + 1
    total, weight = 0.0, 0
    for yy, pp in zip(np.split(y, bounds), np.split(p, bounds), strict=True):
        if 0 < yy.sum() < len(yy):
            total += roc_auc_score(yy, pp) * len(yy)
            weight += len(yy)
    return total / weight if weight else float("nan")


def summary(y: Floats, p: Floats) -> dict[str, float]:
    """Headline metrics for one prediction set."""
    return {
        "n": float(len(y)),
        "ctr": float(y.mean()),
        "ne": normalized_entropy(y, p),
        "logloss": float(logloss_rows(y, p).mean()),
        "auc": float(roc_auc_score(y, p)),
        "calibration_ratio": float(p.mean() / y.mean()),
        "ece": ece(y, p),
    }


def paired_bootstrap(
    diff: Floats, blocks: npt.NDArray[np.int64], resamples: int = 1000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean per-row difference with a 95 % hour-block bootstrap CI (rows in one hour move together)."""
    unique, inverse = np.unique(blocks, return_inverse=True)
    sums = np.bincount(inverse, weights=diff, minlength=len(unique))
    counts = np.bincount(inverse, minlength=len(unique)).astype(np.float64)
    idx = np.random.default_rng(seed).integers(0, len(unique), size=(resamples, len(unique)))
    stats = sums[idx].sum(1) / counts[idx].sum(1)
    low, high = np.quantile(stats, [0.025, 0.975])
    return float(diff.mean()), float(low), float(high)
