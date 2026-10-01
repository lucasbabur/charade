"""Off-policy estimators with hour-block bootstrap intervals.

Inputs, per logged impression i: the target policy's probability of the logged action `pi`, the
logging propensity `mu`, the click `y`, the model's estimate for the logged action `q_logged`, and
the target policy's expected model value `q_pi` = sum_a pi(a|x) q(x, a).

    IPS   = mean(w y),                          w = pi / mu
    SNIPS = sum(w y) / sum(w)
    DR    = mean(q_pi + w (y - q_logged))
    ESS   = (sum w)^2 / sum(w^2)
"""

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel

type Floats = npt.NDArray[np.float64]


class Estimate(BaseModel):
    """One policy x estimator."""

    name: str
    estimator: str
    value: float
    ci_low: float
    ci_high: float
    ess: float
    n: int
    max_weight: float


def _per_block(values: Floats, blocks: npt.NDArray[np.int64]) -> tuple[Floats, int]:
    _, inverse = np.unique(blocks, return_inverse=True)
    return np.bincount(inverse, weights=values), int(inverse.max()) + 1


def estimate(
    name: str,
    pi: Floats,
    mu: Floats,
    y: Floats,
    q_logged: Floats,
    q_pi: Floats,
    blocks: npt.NDArray[np.int64],
    resamples: int = 1000,
    seed: int = 0,
) -> list[Estimate]:
    """IPS, SNIPS and DR for one target policy."""
    w = pi / mu
    ess = float(w.sum() ** 2 / max((w**2).sum(), 1e-12))
    sums = {
        "wy": _per_block(w * y, blocks)[0],
        "w": _per_block(w, blocks)[0],
        "dr": _per_block(q_pi + w * (y - q_logged), blocks)[0],
        "n": _per_block(np.ones_like(y), blocks)[0],
    }
    n_blocks = len(sums["n"])
    idx = np.random.default_rng(seed).integers(0, n_blocks, size=(resamples, n_blocks))
    boot = {k: v[idx].sum(axis=1) for k, v in sums.items()}
    point = {
        "ips": float((w * y).mean()),
        "snips": float((w * y).sum() / w.sum()),
        "dr": float((q_pi + w * (y - q_logged)).mean()),
    }
    draws = {"ips": boot["wy"] / boot["n"], "snips": boot["wy"] / boot["w"], "dr": boot["dr"] / boot["n"]}
    return [
        Estimate(
            name=name,
            estimator=key,
            value=value,
            ci_low=float(np.quantile(draws[key], 0.025)),
            ci_high=float(np.quantile(draws[key], 0.975)),
            ess=ess,
            n=len(y),
            max_weight=float(w.max()),
        )
        for key, value in point.items()
    ]


def lift(
    pi: Floats,
    mu: Floats,
    y: Floats,
    q_logged: Floats,
    q_pi: Floats,
    blocks: npt.NDArray[np.int64],
    resamples: int = 1000,
    seed: int = 0,
) -> dict[str, tuple[float, float, float]]:
    """SNIPS and DR minus the observed logging CTR, with a paired hour-block bootstrap (same resampled hours)."""
    w = pi / mu
    sums = {
        "wy": _per_block(w * y, blocks)[0],
        "w": _per_block(w, blocks)[0],
        "dr": _per_block(q_pi + w * (y - q_logged), blocks)[0],
        "y": _per_block(y, blocks)[0],
        "n": _per_block(np.ones_like(y), blocks)[0],
    }
    idx = np.random.default_rng(seed).integers(0, len(sums["n"]), size=(resamples, len(sums["n"])))
    b = {k: v[idx].sum(axis=1) for k, v in sums.items()}
    observed = b["y"] / b["n"]
    draws = {"snips": b["wy"] / b["w"] - observed, "dr": b["dr"] / b["n"] - observed}
    points = {
        "snips": float((w * y).sum() / w.sum() - y.mean()),
        "dr": float((q_pi + w * (y - q_logged)).mean() - y.mean()),
    }
    return {k: (points[k], float(np.quantile(draws[k], 0.025)), float(np.quantile(draws[k], 0.975))) for k in points}
