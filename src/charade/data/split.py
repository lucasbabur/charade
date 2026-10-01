"""Temporal split. Production predicts the future from the past, so nothing else is honest here."""

from datetime import datetime

import polars as pl

SPLITS = ("train", "val", "test")


def assign_split(
    frame: pl.DataFrame, train_end: datetime, val_end: datetime, test_end: datetime | None = None
) -> pl.DataFrame:
    """Add `split`: train <= train_end < val <= val_end < test <= test_end (inclusive hour starts).

    Rows after `test_end` are dropped, so a partial day in an export never becomes holdout.
    """
    if test_end is not None:
        frame = frame.filter(pl.col("ts") <= test_end)
    return frame.with_columns(
        pl.when(pl.col("ts") <= train_end)
        .then(pl.lit("train"))
        .when(pl.col("ts") <= val_end)
        .then(pl.lit("val"))
        .otherwise(pl.lit("test"))
        .alias("split")
    )
