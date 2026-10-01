from datetime import datetime, timedelta

import pytest

from charade.data import windows
from tests.conftest import FIXTURES, TRAIN_END, VAL_END


def test_rolling_windows_reproduce_the_take_home_split_from_a_complete_export() -> None:
    assert windows.rolling_windows(datetime(2014, 10, 29, 23)) == (TRAIN_END, VAL_END, VAL_END + timedelta(days=1))


def test_a_partial_final_day_is_excluded_from_the_holdout() -> None:
    """The export ends 2014-10-30 05:00: the holdout is 10-29 only, closed by test_end."""
    assert windows.rolling_windows(datetime(2014, 10, 30, 5)) == (TRAIN_END, VAL_END, VAL_END + timedelta(days=1))


def test_cli_prints_env_overrides(capsys: pytest.CaptureFixture[str]) -> None:
    windows.main(str(FIXTURES / "impressions.csv"))
    out = capsys.readouterr().out
    assert out.startswith(
        "export CHARADE_TRAIN_END=2014-10-27T23:00:00 CHARADE_VAL_END=2014-10-28T23:00:00 "
        "CHARADE_TEST_END=2014-10-29T23:00:00"
    )
