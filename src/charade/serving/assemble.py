"""Build model rows for one request: context x N candidates, through the shared feature code.

This is the serving half of train/serve parity: the same raw columns the offline pipeline has,
passed through the same `derive` and `encode`. `charade.analysis.parity` checks they match.
"""

from datetime import datetime

import numpy as np
import polars as pl

from charade.features.counters import RAW_COUNTER_COLUMNS, UserHistory
from charade.features.derive import PLACEHOLDER_DEVICE_ID, derive
from charade.features.spec import Encoded, FeatureSpec, encode
from charade.serving.schemas import RankRequest

CONTEXT_FIELDS = (
    "site_id",
    "site_domain",
    "site_category",
    "app_id",
    "app_domain",
    "app_category",
    "device_id",
    "device_ip",
    "device_model",
    "device_type",
    "device_conn_type",
    "C1",
    "C20",
)
AD_FIELDS = ("banner_pos", "C14", "C15", "C16", "C17", "C18", "C19", "C21")
CHARACTER_FIELDS = ("genre", "safety_tier", "creator_type", "num_interactions", "created_at")


def user_key(device_id: str, device_ip: str, device_model: str) -> str:
    """Same user proxy as `charade.features.derive.user_proxy`."""
    return f"d:{device_id}" if device_id != PLACEHOLDER_DEVICE_ID else f"i:{device_ip}|{device_model}"


def epoch_hour(ts: datetime) -> int:
    """Integer hours since the epoch (UTC-naive)."""
    return int(ts.timestamp() // 3600)


def unknown_character(hour: datetime) -> dict[str, object]:
    """Defaults for a character missing from the table: OOV metadata, brand-safety-strictest tier, age 0."""
    return {
        "genre": "__unknown__",
        "safety_tier": "mature",
        "creator_type": "__unknown__",
        "num_interactions": 0,
        "created_at": hour,
    }


def assemble(
    request: RankRequest, character: dict[str, object], history: UserHistory | None, spec: FeatureSpec
) -> tuple[pl.DataFrame, Encoded]:
    """Derived frame (one row per candidate) and its encoding."""
    n = len(request.candidates)
    counters = (history or UserHistory()).snapshot(epoch_hour(request.hour), [c.C17 for c in request.candidates])
    columns: dict[str, object] = {
        "ts": [request.hour] * n,
        "conversation_turn": [request.conversation_turn] * n,
        "session_msg_count": [request.session_msg_count] * n,
        **{f: [getattr(request, f)] * n for f in CONTEXT_FIELDS},
        **{f: [getattr(c, f) for c in request.candidates] for f in AD_FIELDS},
        **{f: [character[f]] * n for f in CHARACTER_FIELDS},
        **{c: counters[c] for c in RAW_COUNTER_COLUMNS},
    }
    frame = pl.DataFrame(columns).with_columns(
        pl.col("conversation_turn", "session_msg_count").cast(pl.Int32),
        pl.col("num_interactions").cast(pl.Int64),
        pl.col("hours_since_last").fill_nan(None),
        pl.col("ts", "created_at").cast(pl.Datetime("us")),
    )
    derived = derive(frame)
    return derived, encode(spec, derived)


def nan_free(values: np.ndarray) -> bool:
    """True when every score is finite."""
    return bool(np.isfinite(values).all())
