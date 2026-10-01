from datetime import datetime

import pytest

from charade.data import windows
from tests.conftest import FIXTURES, TRAIN_END, VAL_END


def test_rolling_windows_reproduce_the_take_home_split_from_a_complete_export() -> None:
    assert windows.rolling_windows(datetime(2014, 10, 29, 23)) == (TRAIN_END, VAL_END)


def test_a_partial_final_day_is_not_the_holdout_boundary() -> None:
    assert windows.rolling_windows(datetime(2014, 10, 30, 5)) == (TRAIN_END, VAL_END)


def test_cli_prints_env_overrides(capsys: pytest.CaptureFixture[str]) -> None:
    windows.main(str(FIXTURES / "impressions.csv"))
    out = capsys.readouterr().out
    assert out.startswith("export CHARADE_TRAIN_END=2014-10-27T23:00:00 CHARADE_VAL_END=2014-10-28T23:00:00")
