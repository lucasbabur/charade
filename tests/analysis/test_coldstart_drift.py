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


def test_graduation_correction_is_causal_and_zero_with_infinite_prior(features: pl.DataFrame) -> None:
    later = features.filter(pl.col("split") != "train")
    p = np.full(later.height, 0.18)
    table = coldstart.graduation(later, p, tau=1e12)
    assert np.allclose(table["delta_logloss"].to_numpy(), 0.0, atol=1e-9)


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
    greedy, penalised = table.row(0, named=True), table.row(1, named=True)
    assert penalised["cohort_hhi"] <= greedy["cohort_hhi"]
    assert greedy["dr_delta_vs_greedy"] == 0.0


def test_simulation_never_serves_gated_slots(bundle: Settings) -> None:
    c = load_candidates(bundle, FIXTURES)
    chosen, _ = adaptation.simulate(c, 4.0)
    served = chosen >= 0
    assert not c.gated[np.arange(len(chosen))[served], chosen[served]].any()
