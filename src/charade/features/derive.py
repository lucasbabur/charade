"""Columns derived from raw impression + character + counter columns.

Training and serving both call `derive`, on a 1M-row frame or on N candidate rows. That is the
only place features are defined.
"""

import polars as pl

from charade.data.load import PLACEHOLDER_DEVICE_ID
from charade.features.counters import counter_features

APP_SURFACE_SITE_ID = "85f751fd"
"""Rows served inside a host app carry this `site_id`; everything else is a web surface."""

_TURN_EDGES = (1, 2, 3, 5, 10, 20)


def user_proxy() -> pl.Expr:
    """Real `device_id` when present, otherwise `device_ip|device_model` (the placeholder is not a user)."""
    return (
        pl.when(pl.col("device_id") != PLACEHOLDER_DEVICE_ID)
        .then(pl.lit("d:") + pl.col("device_id"))
        .otherwise(pl.lit("i:") + pl.col("device_ip") + pl.lit("|") + pl.col("device_model"))
        .alias("user")
    )


def surface() -> pl.Expr:
    """`app` or `site`."""
    return (
        pl.when(pl.col("site_id") == APP_SURFACE_SITE_ID).then(pl.lit("app")).otherwise(pl.lit("site")).alias("surface")
    )


def _bucket(column: str, edges: tuple[int, ...]) -> pl.Expr:
    labels = [f"<={e}" for e in edges] + [f">{edges[-1]}"]
    return pl.col(column).cut(list(edges), labels=labels, left_closed=False).cast(pl.Utf8)


def derive(frame: pl.DataFrame) -> pl.DataFrame:
    """Add every model input column. Requires raw impression, character and counter columns."""
    frame = frame.with_columns(
        pl.col("ts").dt.hour().cast(pl.Utf8).alias("hour_of_day"),
        pl.when(pl.col("device_id") == PLACEHOLDER_DEVICE_ID)
        .then(pl.lit("__placeholder__"))
        .otherwise(pl.col("device_id"))
        .alias("device_id_real"),
        surface(),
        _bucket("conversation_turn", _TURN_EDGES).alias("turn_bucket"),
        _bucket("session_msg_count", _TURN_EDGES).alias("session_bucket"),
        (pl.col("num_interactions") + 1).log(2).floor().cast(pl.Int32).cast(pl.Utf8).alias("interactions_bucket"),
        (pl.col("num_interactions").cast(pl.Float64) + 1).log().alias("log_num_interactions"),
        ((pl.col("ts") - pl.col("created_at")).dt.total_hours().cast(pl.Float64) / 24 + 1)
        .log()
        .alias("log_character_age_days"),
        (pl.col("conversation_turn").cast(pl.Float64)).log().alias("log_turn"),
        (pl.col("conversation_turn") / pl.col("session_msg_count")).cast(pl.Float64).alias("turn_position"),
    )
    return frame.with_columns(**counter_features(frame))
