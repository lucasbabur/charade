from pathlib import Path

import polars as pl
import pytest

from charade.analysis import fixture
from charade.data.load import load_joined
from charade.features.derive import user_proxy
from tests.conftest import FIXTURES


def test_fixture_keeps_whole_user_histories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fixture, "N_USERS", 50)
    fixture.build(FIXTURES, tmp_path)
    sample = load_joined(tmp_path).with_columns(user_proxy())
    source = load_joined(FIXTURES).with_columns(user_proxy())
    per_user = source.filter(pl.col("user").is_in(sample["user"].unique().implode())).group_by("user").len()
    assert sample.group_by("user").len().sort("user").equals(per_user.sort("user"))
