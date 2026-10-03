import numpy as np
import pytest

from charade.ranking.correction import PairState, decayed, graduated, observe, posterior


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


def test_observe_accumulates_and_decays_by_hours_since_last_update() -> None:
    state = observe(PairState(), hour=10, expected=0.2)
    state = observe(state, hour=10, clicks=1.0)
    assert (state.expected, state.clicks, state.hour) == (pytest.approx(0.2), 1.0, 10)
    later = observe(state, hour=34, expected=0.1, half_life_hours=24.0)
    assert later.expected == pytest.approx(0.1 + 0.1)
    assert later.clicks == pytest.approx(0.5)
    assert later.hour == 34


def test_without_half_life_nothing_decays_and_old_events_do_not_rewind_the_clock() -> None:
    state = observe(PairState(), hour=10, expected=1.0)
    much_later = decayed(state, 1000, None)
    assert (much_later.expected, much_later.clicks) == (1.0, 0.0)
    late = observe(state, hour=5, clicks=1.0)
    assert late.hour == 10
    assert late.clicks == 1.0
