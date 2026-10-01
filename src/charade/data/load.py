"""Load the raw CSVs with an explicit schema and join characters onto impressions.

Every identifier stays a string: hashes have leading zeros and `id` exceeds int64 semantics.
"""

from pathlib import Path

import polars as pl

PLACEHOLDER_DEVICE_ID = "a99f214a"
"""`device_id` value on 82 % of rows; it is a 'device unknown' marker, not a user."""

IMPRESSION_COLUMNS: dict[str, type[pl.DataType]] = {
    "id": pl.Utf8,
    "hour": pl.Utf8,
    "click": pl.Int8,
    "C1": pl.Utf8,
    "banner_pos": pl.Utf8,
    "site_id": pl.Utf8,
    "site_domain": pl.Utf8,
    "site_category": pl.Utf8,
    "app_id": pl.Utf8,
    "app_domain": pl.Utf8,
    "app_category": pl.Utf8,
    "device_id": pl.Utf8,
    "device_ip": pl.Utf8,
    "device_model": pl.Utf8,
    "device_type": pl.Utf8,
    "device_conn_type": pl.Utf8,
    "C14": pl.Utf8,
    "C15": pl.Utf8,
    "C16": pl.Utf8,
    "C17": pl.Utf8,
    "C18": pl.Utf8,
    "C19": pl.Utf8,
    "C20": pl.Utf8,
    "C21": pl.Utf8,
    "character_id": pl.Utf8,
    "conversation_turn": pl.Int32,
    "session_msg_count": pl.Int32,
}

CHARACTER_COLUMNS: dict[str, type[pl.DataType]] = {
    "character_id": pl.Utf8,
    "character_name": pl.Utf8,
    "character_description": pl.Utf8,
    "safety_tier": pl.Utf8,
    "creator_type": pl.Utf8,
    "num_interactions": pl.Int64,
    "created_at": pl.Utf8,
}

SAFETY_TIERS = ("sfw", "suggestive", "mature")


class DataContractError(ValueError):
    """Raw data violates an assumption the pipeline depends on."""


def parse_hour(column: pl.Expr) -> pl.Expr:
    """Parse `YYMMDDHH` into a UTC-naive datetime (chrono needs minutes, so append `00`)."""
    return (column + "00").str.strptime(pl.Datetime("us"), "%y%m%d%H%M", strict=True)


def load_characters(path: Path) -> pl.DataFrame:
    """Characters with parsed creation date and genre (the name prefix, e.g. `romance_1a2b3c` -> `romance`)."""
    frame = _read(path, CHARACTER_COLUMNS)
    frame = frame.with_columns(
        pl.col("created_at").str.strptime(pl.Datetime("us"), "%Y-%m-%d"),
        pl.col("character_name").str.split("_").list.first().alias("genre"),
    )
    _require(frame["character_id"].is_unique().all(), "character_id is not unique")
    _require(frame["safety_tier"].is_in(SAFETY_TIERS).all(), "unknown safety_tier")
    return frame


def load_impressions(path: Path) -> pl.DataFrame:
    """Impressions with parsed `ts`, validated against the invariants features rely on."""
    frame = _read(path, IMPRESSION_COLUMNS).with_columns(parse_hour(pl.col("hour")).alias("ts"))
    _require(frame["id"].is_unique().all(), "duplicate impression ids")
    _require(frame["click"].is_in([0, 1]).all(), "click outside {0,1}")
    _require((frame["conversation_turn"] >= 1).all(), "conversation_turn < 1")
    _require((frame["conversation_turn"] <= frame["session_msg_count"]).all(), "turn > session_msg_count")
    return frame


def load_joined(data_dir: Path) -> pl.DataFrame:
    """Impressions joined with characters, sorted by time then id (deterministic order)."""
    impressions = load_impressions(data_dir / "impressions.csv")
    characters = load_characters(data_dir / "characters.csv")
    joined = impressions.join(characters, on="character_id", how="left", validate="m:1")
    _require(joined["character_name"].null_count() == 0, "impressions reference unknown characters")
    _require((joined["ts"] >= joined["created_at"]).all(), "impression before character creation")
    return joined.sort(["ts", "id"])


def _read(path: Path, columns: dict[str, type[pl.DataType]]) -> pl.DataFrame:
    """Read by column name (never by position) and fail on missing or unexpected columns."""
    header = pl.read_csv(path, n_rows=0).columns
    _require(set(header) == set(columns), f"{path.name}: columns {sorted(set(header) ^ set(columns))} differ")
    return pl.read_csv(path, schema_overrides=columns).select(list(columns))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DataContractError(message)
