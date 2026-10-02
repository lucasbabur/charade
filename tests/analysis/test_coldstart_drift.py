import json
from pathlib import Path

import numpy as np
import polars as pl

from charade.analysis import adaptation, coldstart, drift
from charade.analysis.policy_eval import load_candidates
from charade.config import Settings
from tests.conftest import FIXTURES


def test_coldstart_report(bundle: Settings, tmp_path: Path) -> None:
    tables = coldstart.run(bundle, FIXTURES, tmp_path)
    assert (tmp_path / "coldstart.md").is_file()
    assert all(t.height > 0 for name, t in tables.items() if name.startswith("Permutation"))


def test_beta_prior_recovers_known_spread() -> None:
    rng = np.random.default_rng(0)
    ctr = rng.beta(0.2 * 200, 0.8 * 200, size=400)
    rows = [
        {"character_id": f"c{i}", "genre": "g", "safety_tier": "sfw", "click": int(k)}
        for i, rate in enumerate(ctr)
        for k in rng.binomial(1, rate, size=400)
    ]
    prior = coldstart.beta_prior(pl.DataFrame(rows))
    assert 120 < prior["prior_strength"][0] < 330


def test_drift_writes_contract_and_reports(bundle: Settings, tmp_path: Path) -> None:
    tables = drift.run(bundle, FIXTURES, tmp_path, first_cutoff=5)
    payload = json.loads((bundle.artifacts_dir / "drift.json").read_text())
    assert payload["reference"] == "train"
    assert all(days for days in payload["psi"].values())
    assert tables["Staleness: frozen models scored on later days (1 seed each)"].height > 0


def test_psi_is_zero_for_identical_and_positive_for_shifted() -> None:
    assert drift.psi(np.array([10.0, 20, 30]), np.array([10.0, 20, 30])) == 0.0
    assert drift.psi(np.array([10.0, 20, 30]), np.array([30.0, 20, 10])) > 0.1


def test_adaptation_penalty_reduces_concentration(bundle: Settings, tmp_path: Path) -> None:
    table = adaptation.run(bundle, FIXTURES, tmp_path / "a.md", lambdas=(0.0, 50.0))
    assert "Chosen lambda" in (tmp_path / "a.md").read_text()
    greedy, penalised = table.row(0, named=True), table.row(1, named=True)
    assert penalised["cohort_hhi"] <= greedy["cohort_hhi"]
    assert greedy["dr_delta_vs_greedy"] == 0.0


def test_simulation_never_serves_gated_slots(bundle: Settings) -> None:
    c = load_candidates(bundle, FIXTURES)
    chosen, _ = adaptation.simulate(c, 4.0)
    served = chosen >= 0
    assert not c.gated[np.arange(len(chosen))[served], chosen[served]].any()


def test_lambda_selection_rule_is_non_inferiority() -> None:
    """The lower bound must clear -0.2 pp; a wide interval that merely contains 0 does not qualify."""
    val = pl.DataFrame(
        {
            "lambda": [0.0, 2.0, 4.0, 8.0],
            "dr_delta_vs_greedy": [0.0, -0.0005, -0.0008, 0.0],
            "ci_low": [0.0, -0.0015, -0.0019, -0.009],
            "ci_high": [0.0, 0.0005, 0.0003, 0.009],
        }
    )
    assert adaptation.select(val) == 4.0
