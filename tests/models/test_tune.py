from pathlib import Path

import optuna
import polars as pl
import pytest

from charade.config import DcnConfig, get_settings
from charade.models import tune
from tests.conftest import FIXTURES


def test_tuning_writes_sorted_trials_from_the_inner_split(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tune, "get_settings", lambda: get_settings().model_copy(update={"data_dir": FIXTURES}))
    original = tune.dcn_config

    def small(trial: optuna.Trial) -> DcnConfig:
        return original(trial).model_copy(update={"max_epochs": 1, "hidden": [16], "embedding_dim": 4})

    monkeypatch.setattr(tune, "dcn_config", small)
    tune.run(trials_dcn=2, trials_gbdt=2, out=tmp_path)
    for name in ("dcn", "gbdt"):
        values = pl.read_csv(tmp_path / f"{name}.csv")["value"]
        assert values.len() == 2
        assert values.is_sorted()
    assert get_settings().train_end > tune.INNER_TRAIN_END
