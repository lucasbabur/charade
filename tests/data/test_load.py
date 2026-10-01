import shutil
from pathlib import Path

import polars as pl
import pytest

from charade.data.load import DataContractError, load_impressions, load_joined
from tests.conftest import FIXTURES


def test_columns_are_read_by_name_not_position(joined: pl.DataFrame) -> None:
    assert set(joined["banner_pos"].unique().to_list()) <= {"0", "1", "2", "3", "4", "5", "7"}
    assert joined["C1"].str.len_chars().max() == 4


def test_joined_is_time_ordered_and_has_genre(joined: pl.DataFrame) -> None:
    assert joined["ts"].is_sorted()
    assert joined["genre"].null_count() == 0
    assert {"romance", "horror", "mentor"} <= set(joined["genre"].unique().to_list())


def _copy_fixture(tmp_path: Path) -> Path:
    for name in ("impressions.csv", "characters.csv"):
        shutil.copy(FIXTURES / name, tmp_path / name)
    return tmp_path


def test_reordered_columns_still_load(tmp_path: Path) -> None:
    path = _copy_fixture(tmp_path) / "impressions.csv"
    frame = pl.read_csv(path, infer_schema=False)
    frame.select(list(reversed(frame.columns))).write_csv(path)
    assert load_impressions(path)["click"].dtype == pl.Int8


def test_missing_column_is_rejected(tmp_path: Path) -> None:
    path = _copy_fixture(tmp_path) / "impressions.csv"
    pl.read_csv(path, infer_schema=False).drop("C21").write_csv(path)
    with pytest.raises(DataContractError, match="C21"):
        load_impressions(path)


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    path = _copy_fixture(tmp_path) / "impressions.csv"
    frame = pl.read_csv(path, infer_schema=False)
    pl.concat([frame, frame.head(1)]).write_csv(path)
    with pytest.raises(DataContractError, match="duplicate"):
        load_impressions(path)


def test_unknown_character_is_rejected(tmp_path: Path) -> None:
    root = _copy_fixture(tmp_path)
    characters = pl.read_csv(root / "characters.csv", infer_schema=False)
    characters.slice(1).write_csv(root / "characters.csv")
    with pytest.raises(DataContractError, match="unknown characters"):
        load_joined(root)
