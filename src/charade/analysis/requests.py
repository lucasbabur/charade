"""Turn logged impressions into API requests (used by parity, the load test and sample rankings)."""

from datetime import datetime

import polars as pl

from charade.serving.assemble import AD_FIELDS, CONTEXT_FIELDS
from charade.serving.schemas import RankRequest


def request_from_row(row: dict[str, object], candidates: list[dict[str, object]], request_id: str) -> RankRequest:
    """A RankRequest with the row's context and the given candidates (dicts with AD_FIELDS)."""
    ts = row["ts"]
    assert isinstance(ts, datetime)  # noqa: S101
    payload: dict[str, object] = {
        "request_id": request_id,
        "hour": ts,
        "character_id": str(row["character_id"]),
        "conversation_turn": row["conversation_turn"],
        "session_msg_count": row["session_msg_count"],
        **{f: str(row[f]) for f in CONTEXT_FIELDS},
        "candidates": [
            {
                "candidate_id": str(c.get("candidate_id", f"{c['C14']}@{c['banner_pos']}")),
                **{f: str(c[f]) for f in AD_FIELDS},
            }
            for c in candidates
        ],
    }
    return RankRequest.model_validate(payload)


def creative_pool(frame: pl.DataFrame) -> pl.DataFrame:
    """Distinct creatives (C14 + banner_pos) with their modal ad fields and volume."""
    return (
        frame.group_by("C14", "banner_pos")
        .agg(pl.len().alias("volume"), *[pl.col(f).mode().first() for f in AD_FIELDS if f not in {"C14", "banner_pos"}])
        .sort("volume", "C14", "banner_pos", descending=[True, False, False])
    )
