from pathlib import Path

import pytest

from charade.analysis.experiment import smoke_settings
from charade.models import refit
from tests.conftest import FIXTURES


def test_refit_study_scores_the_validation_day_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(refit, "get_settings", lambda: smoke_settings(tmp_path / "art"))
    report = refit.run(FIXTURES, tmp_path / "refit.json")
    assert "test untouched" in str(report["scored_on"])
    assert float(report["ci_low"]) <= float(report["refit_minus_early_logloss"]) <= float(report["ci_high"])  # type: ignore[arg-type]
