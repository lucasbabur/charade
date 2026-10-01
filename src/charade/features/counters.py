"""User-history counters: strictly causal, identical offline and online.

Definition, for an impression in hour `h` by user `u` on campaign `c` (C17):
    user_imps / user_clicks      impressions / clicks by `u` in hours < h
    user_imps_24h / clicks_24h   the same, restricted to hours h-24 .. h-1
    hours_since_last             h minus the last hour < h with an impression by `u` (null if none)
    user_campaign_imps           impressions of `c` to `u` in hours < h

The current hour is excluded because the order of impressions within an hour is unknown. Including
it creates the well-known Avazu leak: the count for the hour is only known after the hour ends
(docs/hypotheses.md H5). Clicks from earlier hours are assumed to have been joined by serve time.
"""

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt
import polars as pl
from pydantic import BaseModel, Field

RAW_COUNTER_COLUMNS = (
    "user_imps",
    "user_clicks",
    "user_imps_24h",
    "user_clicks_24h",
    "hours_since_last",
    "user_campaign_imps",
)
COUNTER_FEATURES = (
    "user_seen",
    "log_user_imps",
    "log_user_clicks",
    "user_ctr_logit",
    "log_user_imps_24h",
    "log_user_clicks_24h",
    "log_hours_since_last",
    "log_user_campaign_imps",
)
PRIOR_CTR = 0.18
PRIOR_STRENGTH = 4.0
"""Beta(0.72, 3.28) prior for a user's CTR: global rate, worth four impressions of evidence."""


def offline_counters(frame: pl.DataFrame) -> pl.DataFrame:
    """Add RAW_COUNTER_COLUMNS to a frame with `user`, `C17`, `ts`, `click` (any row order)."""
    hourly = (
        frame.group_by(["user", "ts"])
        .agg(pl.len().alias("n"), pl.col("click").cast(pl.Int64).sum().alias("k"))
        .sort(["user", "ts"])
    )
    hourly = hourly.with_columns(
        (pl.col("n").cum_sum().over("user") - pl.col("n")).alias("user_imps"),
        (pl.col("k").cum_sum().over("user") - pl.col("k")).alias("user_clicks"),
        ((pl.col("ts") - pl.col("ts").shift(1).over("user")).dt.total_hours()).alias("hours_since_last"),
    )
    window = hourly.rolling(index_column="ts", period="24h", closed="left", group_by="user").agg(
        pl.col("n").sum().alias("user_imps_24h"), pl.col("k").sum().alias("user_clicks_24h")
    )
    hourly = hourly.join(window, on=["user", "ts"], how="left")
    campaign = (
        frame.group_by(["user", "C17", "ts"])
        .agg(pl.len().alias("n"))
        .sort(["user", "C17", "ts"])
        .with_columns((pl.col("n").cum_sum().over(["user", "C17"]) - pl.col("n")).alias("user_campaign_imps"))
        .drop("n")
    )
    out = frame.join(hourly.drop("n", "k"), on=["user", "ts"], how="left").join(
        campaign, on=["user", "C17", "ts"], how="left"
    )
    return out.with_columns(pl.col("user_imps_24h", "user_clicks_24h").fill_null(0))


def counter_features(frame: pl.DataFrame) -> Mapping[str, pl.Expr]:
    """Model features from raw counters (expressions; evaluated by `derive`)."""
    alpha = PRIOR_CTR * PRIOR_STRENGTH
    rate = (pl.col("user_clicks") + alpha) / (pl.col("user_imps") + PRIOR_STRENGTH)
    del frame
    return {
        "user_seen": (pl.col("user_imps") > 0).cast(pl.Float64),
        "log_user_imps": pl.col("user_imps").cast(pl.Float64).log1p(),
        "log_user_clicks": pl.col("user_clicks").cast(pl.Float64).log1p(),
        "user_ctr_logit": (rate / (1 - rate)).log(),
        "log_user_imps_24h": pl.col("user_imps_24h").cast(pl.Float64).log1p(),
        "log_user_clicks_24h": pl.col("user_clicks_24h").cast(pl.Float64).log1p(),
        "log_hours_since_last": pl.col("hours_since_last").cast(pl.Float64).log1p().fill_null(0.0),
        "log_user_campaign_imps": pl.col("user_campaign_imps").cast(pl.Float64).log1p(),
    }


class UserHistory(BaseModel):
    """Online counter state for one user, the reference for the Redis store (same fields).

    `snapshot(hour, campaigns)` returns what `offline_counters` computes for an impression at
    `hour`. Hours are integer epoch hours. Buckets older than 24 h are pruned, because only totals,
    the last two distinct hours and per-campaign (total, last hour, count in last hour) are needed.
    """

    imps: int = 0
    clicks: int = 0
    last_hour: int | None = None
    prev_hour: int | None = None
    buckets: dict[int, tuple[int, int]] = Field(default_factory=dict[int, tuple[int, int]])
    campaigns: dict[str, tuple[int, int, int]] = Field(default_factory=dict[str, tuple[int, int, int]])

    def record(self, hour: int, campaign: str, click: int) -> None:
        """Add one impression (and its click) at `hour`. Events must arrive in non-decreasing hour."""
        if self.last_hour != hour:
            self.prev_hour, self.last_hour = self.last_hour, hour
        n, k = self.buckets.get(hour, (0, 0))
        self.buckets[hour] = (n + 1, k + click)
        self.buckets = {h: v for h, v in self.buckets.items() if h >= hour - 24}
        self.imps += 1
        self.clicks += click
        total, last, in_last = self.campaigns.get(campaign, (0, -1, 0))
        self.campaigns[campaign] = (total + 1, hour, in_last + 1 if last == hour else 1)

    def record_click(self, hour: int) -> None:
        """Attribute a click that arrived after its impression to the impression's hour."""
        n, k = self.buckets.get(hour, (0, 0))
        if n:
            self.buckets[hour] = (n, k + 1)
        self.clicks += 1

    def snapshot(self, hour: int, campaigns: list[str]) -> dict[str, npt.NDArray[np.float64]]:
        """Raw counters for an impression at `hour`, one row per candidate campaign."""
        current_n, current_k = self.buckets.get(hour, (0, 0))
        earlier = [(n, k) for h, (n, k) in self.buckets.items() if hour - 24 <= h < hour]
        # Strictly earlier hours only; tolerant of out-of-order events (never a negative gap).
        known = [h for h in (self.last_hour, self.prev_hour, *self.buckets) if h is not None and h < hour]
        last = max(known) if known else None
        size = len(campaigns)

        def before(campaign: str) -> int:
            total, last_seen, in_last = self.campaigns.get(campaign, (0, -1, 0))
            return total - (in_last if last_seen == hour else 0)

        return {
            "user_imps": np.full(size, self.imps - current_n, dtype=np.float64),
            "user_clicks": np.full(size, self.clicks - current_k, dtype=np.float64),
            "user_imps_24h": np.full(size, sum(n for n, _ in earlier), dtype=np.float64),
            "user_clicks_24h": np.full(size, sum(k for _, k in earlier), dtype=np.float64),
            "hours_since_last": np.full(size, np.nan if last is None else hour - last, dtype=np.float64),
            "user_campaign_imps": np.array([before(c) for c in campaigns], dtype=np.float64),
        }
