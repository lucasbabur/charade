from datetime import timedelta
from pathlib import Path

import polars as pl

from charade.models.dataset import build_frame
from tests.conftest import FIXTURES, TRAIN_END, VAL_END, fixture_settings


def test_splits_are_disjoint_and_time_ordered(features: pl.DataFrame) -> None:
    bounds = features.group_by("split").agg(pl.col("ts").min().alias("lo"), pl.col("ts").max().alias("hi"))
    by = {r["split"]: r for r in bounds.iter_rows(named=True)}
    assert by["train"]["hi"] <= TRAIN_END < by["val"]["lo"]
    assert by["val"]["hi"] <= VAL_END < by["test"]["lo"]


def test_build_frame_uses_the_settings_it_is_given(tmp_path: Path) -> None:
    """Explicit windows win over the process-wide pyproject settings (an earlier version ignored them)."""
    early = TRAIN_END - timedelta(days=2)
    frame = build_frame(
        fixture_settings(tmp_path).model_copy(update={"train_end": early, "test_end": VAL_END + timedelta(hours=5)}),
        FIXTURES,
    )
    assert frame.filter(pl.col("split") == "train")["ts"].max() <= early  # pyright: ignore[reportOperatorIssue]
    assert frame["ts"].max() <= VAL_END + timedelta(hours=5)  # pyright: ignore[reportOperatorIssue]
