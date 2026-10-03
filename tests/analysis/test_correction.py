from pathlib import Path

import numpy as np
import polars as pl
import pytest

from charade.analysis import correction
from charade.analysis.policy_eval import load_candidates
from charade.config import Settings
from charade.ranking.correction import posterior
from tests.conftest import FIXTURES


def test_replay_never_serves_gated_slots_and_scores_with_earlier_hours_only(bundle: Settings) -> None:
    c = load_candidates(bundle, FIXTURES)
    r = correction.replay(c, prior=20.0, half_life=None, z=0.0)
    served = r["chosen"] >= 0
    assert not c.gated[np.arange(len(c.y))[served], r["chosen"][served]].any()
    first_hour = c.rows["ts"].to_numpy().astype("datetime64[h]")
    inside = c.logged >= 0
    opening = inside & (first_hour == first_hour.min())
    # Nothing has been observed before the first hour, so its corrected pCTR equals the model's.
    assert np.allclose(r["p_adj"][opening], c.at_logged(c.pctr)[opening])
    assert not r["graduated"][opening].any()


def test_replay_with_infinite_prior_is_the_greedy_policy(bundle: Settings) -> None:
    c = load_candidates(bundle, FIXTURES)
    r = correction.replay(c, prior=1e12, half_life=None, z=0.0)
    assert (r["chosen"] == correction.greedy_slots(c)).all()
    assert correction.dr_rows(c, r["chosen"]).mean() == correction.dr_rows(c, correction.greedy_slots(c)).mean()


def test_replay_state_matches_a_direct_recount(bundle: Settings) -> None:
    """The vectorised sums equal a plain per-row recount of earlier-hour logged outcomes."""
    c = load_candidates(bundle, FIXTURES)
    prior = 5.0
    r = correction.replay(c, prior, None, 0.0)
    hours = c.rows["ts"].to_numpy().astype("datetime64[h]").astype(np.int64)
    genres = c.rows["genre"].to_list()
    inside = np.flatnonzero(c.logged >= 0)
    for t in inside[:50]:
        key = (str(c.campaign[t, c.logged[t]]), genres[t])
        earlier = [u for u in inside if hours[u] < hours[t] and (str(c.campaign[u, c.logged[u]]), genres[u]) == key]
        expected = sum(c.pctr[u, c.logged[u]] for u in earlier)
        clicks = sum(c.y[u] for u in earlier)
        mean, _ = posterior(np.array([expected]), np.array([clicks]), prior)
        assert r["p_adj"][t] == pytest.approx(min(c.pctr[t, c.logged[t]] * mean[0], 1.0))


def test_selection_rule_is_non_inferiority_on_deterministic_rows() -> None:
    val = pl.DataFrame(
        {
            "prior": [5.0, 20.0, 80.0, 80.0],
            "half_life_h": [float("inf"), 12.0, float("inf"), 48.0],
            "z": [0.0, 0.0, 0.0, 1.0],
            "dr_delta_vs_greedy_pp": [0.9, 0.5, 0.4, 2.0],
            "ci_low_pp": [-0.9, -0.1, -0.15, 1.0],
        }
    )
    assert correction.select(val) == (20.0, 12.0)
    assert correction.select(val.filter(pl.col("prior") == 5.0))[1] is None


def test_run_writes_report_json_and_graduation_curve(bundle: Settings, tmp_path: Path) -> None:
    out = correction.run(bundle, FIXTURES, tmp_path / "correction.md", tmp_path / "graduation.csv", priors=(20.0,))
    text = (tmp_path / "correction.md").read_text()
    assert "## Validation" in text
    assert "## Test" in text
    assert (tmp_path / "correction.json").is_file()
    assert out["val"].height == len(correction.HALF_LIVES) * len(correction.ZS)
    assert {"hour", "graduated", "rows"} <= set(out["graduation"].columns) or out["graduation"].is_empty()
