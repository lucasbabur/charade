import json
from pathlib import Path

import numpy as np
import polars as pl

from charade.analysis import policy_eval
from charade.config import Settings
from charade.evaluation.ope import estimate, lift


def _synthetic(
    seed: int, n: int = 40_000
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Two actions, logging picks action 1 w.p. 0.3; true CTRs 0.1 and 0.3; target always picks 1."""
    rng = np.random.default_rng(seed)
    action = (rng.random(n) < 0.3).astype(int)
    y = rng.binomial(1, np.where(action == 1, 0.3, 0.1)).astype(np.float64)
    mu = np.where(action == 1, 0.3, 0.7)
    pi = (action == 1).astype(np.float64)
    q = np.where(action == 1, 0.28, 0.12)
    return pi, mu, y, q, np.full(n, 0.28), np.arange(n) // 400


def test_estimators_recover_the_target_policy_value() -> None:
    values = {e.estimator: e for e in estimate("always-1", *_synthetic(0))}
    for key in ("ips", "snips", "dr"):
        assert values[key].ci_low < 0.3 < values[key].ci_high
    assert values["snips"].ess < 40_000


def test_lift_is_positive_when_target_beats_logging() -> None:
    result = lift(*_synthetic(1))
    for _, low, _ in result.values():
        assert low > 0


def test_policy_evaluation_writes_ope_and_gate_respecting_decisions(bundle: Settings, tmp_path: Path) -> None:
    results = policy_eval.run(bundle, report=tmp_path / "ope.md")
    names = {r.name for r in results}
    assert {"logging (observed)", "greedy pCTR (no gates)", "shipped policy (gates + 5% exploration)"} <= names
    decisions = [json.loads(line) for line in (bundle.artifacts_dir / "decisions.jsonl").read_text().splitlines()]
    assert decisions
    for d in decisions:
        gated = {c["candidate_id"] for c in d["candidates"] if c["gated"]}
        assert d["chosen_id"] not in gated
        if d["chosen_id"] is not None:
            assert abs(sum(c["propensity"] for c in d["candidates"]) - 1) < 1e-9
    assert "Lift over the logging policy" in (tmp_path / "ope.md").read_text()


def test_candidate_sets_keep_only_multi_candidate_cells(bundle: Settings) -> None:
    frame = pl.DataFrame(
        {
            "site_id": ["s"] * 5,
            "app_id": ["a"] * 5,
            "ts": [0] * 5,
            "C14": ["x", "x", "y", "z", "z"],
            "banner_pos": ["0"] * 5,
            **{f: ["f"] * 5 for f in ("C15", "C16", "C17", "C18", "C19", "C21")},
        }
    )
    served, rows = policy_eval.candidate_sets(frame)
    assert served["mu"].sum() == 1.0
    assert rows.height == 5
    assert set(served["k"].to_list()) == {3}
    del bundle
