from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from charade.data.load import load_joined
from charade.data.split import assign_split
from charade.features.counters import offline_counters
from charade.features.derive import derive, user_proxy

FIXTURES = Path(__file__).parent / "fixtures"
TRAIN_END = datetime(2014, 10, 27, 23)
VAL_END = datetime(2014, 10, 28, 23)


@pytest.fixture(scope="session")
def joined() -> pl.DataFrame:
    return load_joined(FIXTURES)


@pytest.fixture(scope="session")
def features(joined: pl.DataFrame) -> pl.DataFrame:
    frame = derive(offline_counters(joined.with_columns(user_proxy())))
    return assign_split(frame, TRAIN_END, VAL_END)
