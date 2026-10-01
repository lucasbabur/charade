"""Evaluation slices (attached to every prediction row as `slice_<name>`)."""

import polars as pl

SUPPORT_EDGES = (0, 20, 200)


def add_slices(frame: pl.DataFrame) -> pl.DataFrame:
    """Add slice columns. `character_support` = the character's impressions in the training split."""
    support = frame.filter(pl.col("split") == "train").group_by("character_id").len().rename({"len": "train_imps"})
    return (
        frame.join(support, on="character_id", how="left")
        .with_columns(pl.col("train_imps").fill_null(0))
        .with_columns(
            pl.when(pl.col("train_imps") == 0)
            .then(pl.lit("0 (cold)"))
            .when(pl.col("train_imps") <= SUPPORT_EDGES[1])
            .then(pl.lit("1-20"))
            .when(pl.col("train_imps") <= SUPPORT_EDGES[2])
            .then(pl.lit("21-200"))
            .otherwise(pl.lit(">200"))
            .alias("slice_character_support"),
            pl.col("genre").alias("slice_genre"),
            pl.col("safety_tier").alias("slice_safety_tier"),
            pl.col("surface").alias("slice_surface"),
            pl.col("ts").dt.date().cast(pl.Utf8).alias("slice_day"),
            pl.when(pl.col("user_imps") > 0).then(pl.lit("seen")).otherwise(pl.lit("new")).alias("slice_user_seen"),
            pl.col("banner_pos").alias("slice_banner_pos"),
        )
    )


SLICES = ("character_support", "genre", "safety_tier", "surface", "day", "user_seen", "banner_pos")
