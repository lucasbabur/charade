import random
from datetime import datetime

import numpy as np
import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st

from charade.features.counters import RAW_COUNTER_COLUMNS, WINDOW_HOURS, UserHistory, offline_counters


def _epoch_hour(ts: datetime) -> int:
    return int(ts.timestamp() // 3600)


def _replay(frame: pl.DataFrame) -> int:
    """Replay events through the online state and count mismatches with the offline columns."""
    histories: dict[str, UserHistory] = {}
    mismatches = 0
    for row in frame.sort(["ts", "id"]).iter_rows(named=True):
        history = histories.setdefault(row["user"], UserHistory())
        hour = _epoch_hour(row["ts"])
        online = history.snapshot(hour, [row["C17"]])
        for column in RAW_COUNTER_COLUMNS:
            expected = np.nan if row[column] is None else float(row[column])
            got = online[column][0]
            mismatches += not (got == expected or (np.isnan(got) and np.isnan(expected)))
        history.record(hour, row["C17"], row["click"])
    return mismatches


def test_online_state_matches_offline_counters_on_fixture(features: pl.DataFrame) -> None:
    assert _replay(features) == 0


def test_counters_exclude_the_current_hour(features: pl.DataFrame) -> None:
    first_hour = features.group_by("user").agg(pl.col("ts").min().alias("first"))
    rows = features.join(first_hour, on="user").filter(pl.col("ts") == pl.col("first"))
    assert rows["user_imps"].max() == 0
    assert rows["hours_since_last"].null_count() == rows.height


events = st.lists(
    st.tuples(st.integers(0, 60), st.sampled_from(["u1", "u2"]), st.sampled_from(["c1", "c2"]), st.integers(0, 1)),
    min_size=1,
    max_size=60,
)


@settings(max_examples=200, deadline=None)
@given(events)
def test_online_matches_offline_for_any_event_sequence(raw: list[tuple[int, str, str, int]]) -> None:
    base = datetime(2014, 10, 21)
    frame = pl.DataFrame(
        {
            "id": [str(i) for i in range(len(raw))],
            "ts": [base.replace(day=21 + h // 24, hour=h % 24) for h, *_ in raw],
            "user": [u for _, u, _, _ in raw],
            "C17": [c for *_, c, _ in raw],
            "click": [k for *_, k in raw],
        }
    )
    assert _replay(offline_counters(frame)) == 0


def test_out_of_order_events_never_produce_negative_gaps() -> None:
    history = UserHistory()
    history.record(100, "c", 0)
    history.record(120, "c", 1)
    snapshot = history.snapshot(110, ["c"])
    assert snapshot["hours_since_last"][0] == 10
    assert np.isnan(history.snapshot(90, ["c"])["hours_since_last"][0])


@settings(max_examples=300, deadline=None)
@given(events, st.randoms(use_true_random=False), st.integers(0, 70))
def test_snapshot_is_independent_of_arrival_order(
    raw: list[tuple[int, str, str, int]], shuffle: random.Random, query_hour: int
) -> None:
    """Record events in any order; a snapshot at hour h equals the offline counters over events before h."""
    arrival = list(raw)
    shuffle.shuffle(arrival)
    histories: dict[str, UserHistory] = {}
    for hour, user, campaign, click in arrival:
        histories.setdefault(user, UserHistory()).record(hour, campaign, click)
    for user, history in histories.items():
        assert history.newest is not None
        if query_hour < history.newest - (WINDOW_HOURS - 24):
            continue  # outside the guarantee: request more than 24 h behind this user's newest event
        prior = [(h, c, k) for h, u, c, k in raw if u == user and h < query_hour]
        got = history.snapshot(query_hour, ["c1", "c2"])
        assert got["user_imps"][0] == len(prior)
        assert got["user_clicks"][0] == sum(k for *_, k in prior)
        assert got["user_imps_24h"][0] == sum(1 for h, *_ in prior if h >= query_hour - 24)
        assert list(got["user_campaign_imps"]) == [sum(1 for _, c, _ in prior if c == x) for x in ("c1", "c2")]
        expected_gap = query_hour - max(h for h, *_ in prior) if prior else None
        gap = got["hours_since_last"][0]
        assert (np.isnan(gap) and expected_gap is None) or gap == expected_gap


def test_future_events_do_not_leak_into_an_earlier_snapshot() -> None:
    """The reviewer's reproduction: an event at 101 recorded before one at 100."""
    history = UserHistory()
    history.record(101, "c", 0)
    history.record(100, "c", 0)
    assert history.snapshot(100, ["c"])["user_imps"][0] == 0
    assert history.snapshot(101, ["c"])["user_imps"][0] == 1


def test_late_click_is_attributed_to_its_impression_hour() -> None:
    history = UserHistory()
    history.record(100, "c", 0)
    history.record(103, "c", 0)
    history.record_click(100)
    assert history.snapshot(101, ["c"])["user_clicks"][0] == 1
    assert history.snapshot(100, ["c"])["user_clicks"][0] == 0


def test_cap_exposures_include_the_current_hour_but_the_feature_does_not() -> None:
    history = UserHistory()
    for _ in range(9):
        history.record(100, "c", 0)
    assert history.snapshot(100, ["c"])["user_campaign_imps"][0] == 0
    assert history.exposures_so_far(["c"])[0] == 9
