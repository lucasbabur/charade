from datetime import datetime

import numpy as np
import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st

from charade.features.counters import RAW_COUNTER_COLUMNS, UserHistory, offline_counters


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
