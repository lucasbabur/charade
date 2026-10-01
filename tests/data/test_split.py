import polars as pl

from tests.conftest import TRAIN_END, VAL_END


def test_splits_are_disjoint_and_time_ordered(features: pl.DataFrame) -> None:
    bounds = features.group_by("split").agg(pl.col("ts").min().alias("lo"), pl.col("ts").max().alias("hi"))
    by = {r["split"]: r for r in bounds.iter_rows(named=True)}
    assert by["train"]["hi"] <= TRAIN_END < by["val"]["lo"]
    assert by["val"]["hi"] <= VAL_END < by["test"]["lo"]
