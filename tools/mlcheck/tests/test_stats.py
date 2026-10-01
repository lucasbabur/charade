import math

import numpy as np
import pytest

from mlcheck import stats


def test_log_loss_matches_closed_form() -> None:
    y = np.array([1.0, 0.0])
    p = np.array([0.8, 0.3])
    assert stats.log_loss(y, p) == pytest.approx(-(math.log(0.8) + math.log(0.7)) / 2)


def test_constant_base_rate_predictor_has_unit_normalized_entropy() -> None:
    y = np.array([1.0, 0.0, 0.0, 0.0])
    assert stats.normalized_entropy(y, np.full(4, 0.25)) == pytest.approx(1.0)


def test_entropy_degenerate_rates() -> None:
    assert stats.entropy(0.0) == 0.0
    assert stats.entropy(1.0) == 0.0


def test_calibration_ratio_and_ece_of_perfect_predictions() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.6, 200_000)
    y = rng.binomial(1, p).astype(np.float64)
    assert stats.calibration_ratio(y, p) == pytest.approx(1.0, abs=0.01)
    assert stats.expected_calibration_error(y, p, 15) < 0.01


def test_ece_detects_overconfidence() -> None:
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.6, 100_000)
    y = rng.binomial(1, p).astype(np.float64)
    assert stats.expected_calibration_error(y, np.clip(p * 1.5, 0, 0.99), 15) > 0.05


def test_block_bootstrap_covers_true_mean() -> None:
    rng = np.random.default_rng(2)
    covered = 0
    trials = 200
    for _ in range(trials):
        block_effect = np.repeat(rng.normal(0, 1, 40), 50)
        diff = block_effect + rng.normal(0, 1, 2000)
        _, low, high = stats.paired_block_bootstrap(diff, np.repeat(np.arange(40), 50), 400, 0.9, seed=3)
        covered += low <= 0 <= high
    assert 0.8 <= covered / trials <= 0.97


def test_block_bootstrap_is_wider_than_naive_under_block_correlation() -> None:
    rng = np.random.default_rng(4)
    diff = np.repeat(rng.normal(0, 1, 30), 100) + rng.normal(0, 0.1, 3000)
    _, low, high = stats.paired_block_bootstrap(diff, np.repeat(np.arange(30), 100), 1000, 0.95)
    naive_half_width = 1.96 * diff.std() / math.sqrt(len(diff))
    assert (high - low) / 2 > 5 * naive_half_width
