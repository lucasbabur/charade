import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from charade.analysis import policy_eval
from charade.config import Settings
from charade.evaluation.metrics import normalized_entropy
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
    assert policy_eval.GATED in names
    assert "What the shipped policy gives up" in (tmp_path / "ope.md").read_text()
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


def test_rows_whose_logged_ad_is_outside_the_top_k_stay_in_the_population() -> None:
    """Eleven equally frequent ads, all with CTR 0.2: a uniform policy over the top ten is worth 0.2.

    Dropping the eleventh ad's rows while keeping its unconditional share (an earlier version) gave 0.22.
    """
    n_ads, per_ad = 11, 10
    frame = pl.DataFrame(
        {
            "site_id": ["s"] * n_ads * per_ad,
            "app_id": ["a"] * n_ads * per_ad,
            "ts": [0] * n_ads * per_ad,
            "C14": [f"c{i:02d}" for i in range(n_ads) for _ in range(per_ad)],
            "banner_pos": ["0"] * n_ads * per_ad,
            "click": [int(j < 2) for _ in range(n_ads) for j in range(per_ad)],
            **{f: ["f"] * n_ads * per_ad for f in ("C15", "C16", "C17", "C18", "C19", "C21")},
        }
    )
    served, rows = policy_eval.candidate_sets(frame)
    assert served.height == policy_eval.TOP_K
    assert rows.height == n_ads * per_ad
    assert (rows["logged_slot"] == -1).sum() == per_ad
    logged = rows["logged_slot"].to_numpy()
    pi = np.where(logged >= 0, 1 / policy_eval.TOP_K, 0.0)
    y = rows["click"].to_numpy().astype(np.float64)
    zeros = np.zeros_like(y)
    ips = estimate("uniform top-k", pi, rows["logged_mu"].to_numpy(), y, zeros, zeros, np.zeros(len(y), np.int64))[0]
    assert ips.value == pytest.approx(0.2)


def test_opportunity_diagnostic_flattens_within_candidate_sets(bundle: Settings, tmp_path: Path) -> None:
    from charade.analysis import opportunity  # noqa: PLC0415

    report = opportunity.run(bundle, out=tmp_path / "opportunity.json")
    assert report["opportunities"] != 0
    assert report["mean_candidates"] >= 2  # pyright: ignore[reportOperatorIssue]
    cost = report["logloss_cost_of_flattening"]
    assert cost["ci_low"] <= cost["delta"] <= cost["ci_high"]  # pyright: ignore[reportIndexIssue]


def test_flattening_excludes_missing_logged_ads_and_ignores_padding(bundle: Settings) -> None:
    from charade.analysis import opportunity  # noqa: PLC0415

    candidates = replace(
        policy_eval.load_candidates(bundle),
        logged=np.array([0, 1, -1, 0]),
        y=np.array([0.0, 1.0, 1.0, 1.0]),
        blocks=np.arange(4),
        mask=np.array([[True, True, False]] * 4),
        pctr=np.array([[0.1, 0.3, 99], [0.2, 0.6, 99], [0.3, 0.9, 99], [0.3, 0.5, 99]]),
    )
    report = opportunity.compare(candidates)
    assert report["opportunities"] == 3
    assert report["excluded_outside_candidate_set"] == 1
    assert report["flattened"] == {
        "ne": pytest.approx(normalized_entropy(np.array([0.0, 1.0, 1.0]), np.array([0.2, 0.4, 0.4]))),
        "auc": 1.0,
    }
    assert "share_of_skill_that_ranks_ads" not in report


def test_flattening_loss_can_change_without_changing_the_selected_ad(bundle: Settings) -> None:
    """Duplicating a lower-scored alternative changes the mean, not the winner or logged predictions."""
    from charade.analysis import opportunity  # noqa: PLC0415

    candidates = replace(
        policy_eval.load_candidates(bundle),
        logged=np.zeros(4, dtype=np.int64),
        y=np.array([0.0, 1.0, 0.0, 1.0]),
        blocks=np.arange(4),
        mask=np.ones((4, 2), dtype=bool),
        pctr=np.array([[0.5, 0.1]] * 4),
    )
    duplicated = replace(candidates, mask=np.ones((4, 3), dtype=bool), pctr=np.array([[0.5, 0.1, 0.1]] * 4))
    before, after = opportunity.compare(candidates), opportunity.compare(duplicated)
    assert np.array_equal(candidates.pctr.argmax(axis=1), duplicated.pctr.argmax(axis=1))
    assert before["model"] == after["model"]
    assert before["logloss_cost_of_flattening"] != after["logloss_cost_of_flattening"]
