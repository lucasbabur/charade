"""Metrics computed independently from the project's code, so a buggy report cannot pass its own gate."""

import math

import numpy as np
import numpy.typing as npt

type Floats = npt.NDArray[np.float64]

_EPS = 1e-15


def log_loss_per_row(y: Floats, p: Floats) -> Floats:
    """Binary cross-entropy per row, with probabilities clipped away from 0 and 1."""
    clipped = np.clip(p, _EPS, 1 - _EPS)
    return -(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))


def log_loss(y: Floats, p: Floats) -> float:
    """Mean binary cross-entropy."""
    return float(log_loss_per_row(y, p).mean())


def entropy(rate: float) -> float:
    """Entropy of a Bernoulli(rate), in nats."""
    if rate <= 0 or rate >= 1:
        return 0.0
    return -(rate * math.log(rate) + (1 - rate) * math.log(1 - rate))


def normalized_entropy(y: Floats, p: Floats) -> float:
    """Log loss divided by the entropy of the evaluation set's base rate; < 1 beats the constant predictor."""
    base = entropy(float(y.mean()))
    return math.inf if base == 0 else log_loss(y, p) / base


def calibration_ratio(y: Floats, p: Floats) -> float:
    """Mean prediction over observed rate; 1.0 is calibrated in aggregate."""
    observed = float(y.mean())
    return math.inf if observed == 0 else float(p.mean()) / observed


def expected_calibration_error(y: Floats, p: Floats, bins: int) -> float:
    """ECE with equal-mass bins (robust when predictions concentrate near the base rate)."""
    order = np.argsort(p, kind="stable")
    total = len(p)
    ece = 0.0
    for chunk in np.array_split(order, bins):
        if len(chunk) == 0:
            continue
        ece += len(chunk) / total * abs(float(p[chunk].mean()) - float(y[chunk].mean()))
    return ece


def paired_block_bootstrap(
    row_diff: Floats, blocks: npt.NDArray[np.int64], resamples: int, confidence: float, seed: int = 0
) -> tuple[float, float, float]:
    """Mean of a per-row paired difference with a block-bootstrap percentile CI.

    Rows in the same block (e.g. the same hour) are resampled together, because errors within an
    hour are correlated; resampling rows independently would make the interval too narrow.

    Args:
        row_diff: Per-row difference (e.g. model loss minus baseline loss).
        blocks: Block id per row.
        resamples: Number of bootstrap resamples.
        confidence: Interval coverage, e.g. 0.95.
        seed: RNG seed.

    Returns:
        Point estimate, lower bound, upper bound.
    """
    unique, inverse = np.unique(blocks, return_inverse=True)
    sums = np.bincount(inverse, weights=row_diff, minlength=len(unique))
    counts = np.bincount(inverse, minlength=len(unique)).astype(np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(unique), size=(resamples, len(unique)))
    stats = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    alpha = (1 - confidence) / 2
    low, high = np.quantile(stats, [alpha, 1 - alpha])
    return float(row_diff.mean()), float(low), float(high)
