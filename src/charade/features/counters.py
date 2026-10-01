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


WINDOW_HOURS = 48
"""Hourly detail kept per user, measured back from the newest event. Snapshots for hours within 24 h of
the newest event are exact whatever the arrival order; older requests (rare) see pruned detail."""


class UserHistory(BaseModel):
    """Online counter state for one user (the Redis store persists exactly this).

    `snapshot(hour, campaigns)` returns what `offline_counters` computes for an impression at `hour`:
    only events from strictly earlier hours count, whatever order the events arrived in. Totals cover
    all time; hourly buckets (overall and per campaign) cover the last `WINDOW_HOURS`, so events from
    the current or later hours can be subtracted and the 24 h window summed.
    """

    imps: int = 0
    clicks: int = 0
    newest: int | None = None
    pruned_last: int | None = None
    """Latest hour among buckets already pruned (for hours-since-last beyond the window)."""
    buckets: dict[int, tuple[int, int]] = Field(default_factory=dict[int, tuple[int, int]])
    campaign_totals: dict[str, int] = Field(default_factory=dict[str, int])
    campaign_buckets: dict[str, dict[int, int]] = Field(default_factory=dict[str, dict[int, int]])

    def record(self, hour: int, campaign: str, click: int = 0) -> None:
        """Add one impression at `hour` (any arrival order)."""
        self.imps += 1
        self.clicks += click
        self.campaign_totals[campaign] = self.campaign_totals.get(campaign, 0) + 1
        self.newest = hour if self.newest is None else max(self.newest, hour)
        if hour < self.newest - WINDOW_HOURS:
            self.pruned_last = hour if self.pruned_last is None else max(self.pruned_last, hour)
            return
        n, k = self.buckets.get(hour, (0, 0))
        self.buckets[hour] = (n + 1, k + click)
        per_hour = self.campaign_buckets.setdefault(campaign, {})
        per_hour[hour] = per_hour.get(hour, 0) + 1
        self._prune()

    def record_click(self, hour: int) -> None:
        """Attribute a click that arrived after its impression to the impression's hour."""
        self.clicks += 1
        if hour in self.buckets:
            n, k = self.buckets[hour]
            self.buckets[hour] = (n, k + 1)

    def _prune(self) -> None:
        assert self.newest is not None  # noqa: S101 - set by record() before pruning
        cutoff = self.newest - WINDOW_HOURS
        old = [h for h in self.buckets if h < cutoff]
        if old:
            self.pruned_last = max([*old, *([self.pruned_last] if self.pruned_last is not None else [])])
            for h in old:
                del self.buckets[h]
        for campaign, per_hour in self.campaign_buckets.items():
            self.campaign_buckets[campaign] = {h: n for h, n in per_hour.items() if h >= cutoff}

    def exposures_so_far(self, campaigns: list[str]) -> npt.NDArray[np.float64]:
        """Every recorded impression per campaign, including the current hour and any arrival order.

        Use this for the frequency cap (a serving constraint), never as a model feature: the model's
        `user_campaign_imps` must exclude the current hour (unknown at training time, see H5).
        """
        return np.array([self.campaign_totals.get(c, 0) for c in campaigns], dtype=np.float64)

    def snapshot(self, hour: int, campaigns: list[str]) -> dict[str, npt.NDArray[np.float64]]:
        """Raw counters for an impression at `hour`, one row per candidate campaign."""
        later_n = sum(n for h, (n, _) in self.buckets.items() if h >= hour)
        later_k = sum(k for h, (_, k) in self.buckets.items() if h >= hour)
        window = [(n, k) for h, (n, k) in self.buckets.items() if hour - 24 <= h < hour]
        earlier = [h for h in self.buckets if h < hour]
        if earlier:
            last: int | None = max(earlier)
        else:
            last = self.pruned_last if self.pruned_last is not None and self.pruned_last < hour else None
        size = len(campaigns)

        def before(campaign: str) -> int:
            later = sum(n for h, n in self.campaign_buckets.get(campaign, {}).items() if h >= hour)
            return self.campaign_totals.get(campaign, 0) - later

        return {
            "user_imps": np.full(size, self.imps - later_n, dtype=np.float64),
            "user_clicks": np.full(size, self.clicks - later_k, dtype=np.float64),
            "user_imps_24h": np.full(size, sum(n for n, _ in window), dtype=np.float64),
            "user_clicks_24h": np.full(size, sum(k for _, k in window), dtype=np.float64),
            "hours_since_last": np.full(size, np.nan if last is None else hour - last, dtype=np.float64),
            "user_campaign_imps": np.array([before(c) for c in campaigns], dtype=np.float64),
        }
