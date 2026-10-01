"""Rolling split windows for production retraining (`python -m charade.data.windows <impressions.csv>`).

The take-home pins its split dates in `[tool.charade]`. A daily retrain instead derives them from
the export: the last complete day is the holdout (scored once by the gates), the day before is
validation (early stopping, calibration), everything earlier is training. Printed as shell exports
that override the pyproject values through `CHARADE_*` environment variables.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from charade.data.load import parse_hour


def rolling_windows(last_hour: datetime) -> tuple[datetime, datetime]:
    """(train_end, val_end) as inclusive hour starts, given the newest hour in the export.

    A partial final day is dropped from the holdout's boundary computation: the holdout is the last
    day whose 23:00 hour is present, so the windows are always whole days.
    """
    last_complete_day = last_hour.date() if last_hour.hour == 23 else last_hour.date() - timedelta(days=1)
    val_day = last_complete_day - timedelta(days=1)
    val_end = datetime.combine(val_day, datetime.min.time()) + timedelta(hours=23)
    return val_end - timedelta(days=1), val_end


def main(path: str) -> None:
    """Print `export CHARADE_TRAIN_END=... CHARADE_VAL_END=...` for the given export."""
    last = pl.scan_csv(Path(path), infer_schema=False).select(parse_hour(pl.col("hour")).max()).collect().item()
    train_end, val_end = rolling_windows(last)
    print(f"export CHARADE_TRAIN_END={train_end.isoformat()} CHARADE_VAL_END={val_end.isoformat()}")


if __name__ == "__main__":
    main(sys.argv[1])
