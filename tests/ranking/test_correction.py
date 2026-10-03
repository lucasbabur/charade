import numpy as np
import pytest

from charade.ranking.correction import graduated, posterior


def test_empty_pair_trusts_the_model_with_prior_width() -> None:
    mean, sd = posterior(np.array([0.0]), np.array([0.0]), prior=80.0)
    assert mean[0] == 1.0
    assert sd[0] == pytest.approx(1 / np.sqrt(80.0))


def test_posterior_moves_toward_the_observed_rate_and_narrows() -> None:
    expected, clicks = np.array([200.0]), np.array([300.0])
    mean, sd = posterior(expected, clicks, prior=80.0)
    assert 1.0 < mean[0] < 1.5
    assert sd[0] < 1 / np.sqrt(80.0)
    assert posterior(expected * 10, clicks * 10, prior=80.0)[0][0] == pytest.approx(1.5, abs=0.05)


def test_graduation_is_when_logged_evidence_outweighs_the_prior() -> None:
    assert graduated(np.array([79.9, 80.0, 500.0]), prior=80.0).tolist() == [False, True, True]
