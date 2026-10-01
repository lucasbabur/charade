"""Turn serving logs into training rows (`python -m charade.data.events <log.jsonl> <out.csv>`).

The API writes three JSON events (CloudWatch -> Firehose -> S3 in production):
    decision    request_id, hour, character_id, context (request fields), ads (per-candidate attributes)
    impression  impression_id, request_id, candidate_id, hour   (once per impression_id)
    click       impression_id                                   (once per impression_id, may arrive late)

Each impression joins its decision on request_id and its served candidate's attributes on
candidate_id; click = 1 if a click event exists for the impression. The result has exactly the
columns of `impressions.csv`, so the training pipeline consumes it unchanged. Impressions whose
decision is missing (log loss, retention) are dropped and counted, not guessed.
"""

import json
import sys
from collections.abc import Iterable
from pathlib import Path

import polars as pl

from charade.data.load import IMPRESSION_COLUMNS

AD_FIELDS = ("banner_pos", "C14", "C15", "C16", "C17", "C18", "C19", "C21")


def _records(lines: Iterable[str]) -> Iterable[dict[str, object]]:
    for raw in lines:
        line = raw.strip()
        if line.startswith("{"):
            record = json.loads(line)
            if record.get("event") in {"decision", "impression", "click"}:
                yield record


def build_rows(lines: Iterable[str]) -> tuple[pl.DataFrame, int]:
    """(training rows in the impressions.csv schema, impressions dropped for a missing decision)."""
    decisions: dict[str, dict[str, object]] = {}
    impressions: list[dict[str, object]] = []
    clicked: set[str] = set()
    for record in _records(lines):
        if record["event"] == "decision":
            decisions[str(record["request_id"])] = record
        elif record["event"] == "impression":
            impressions.append(record)
        else:
            clicked.add(str(record["impression_id"]))
    rows: list[dict[str, object]] = []
    dropped = 0
    for imp in impressions:
        decision = decisions.get(str(imp["request_id"]))
        ads = decision.get("ads") if decision else None
        if not isinstance(ads, dict) or str(imp["candidate_id"]) not in ads:
            dropped += 1
            continue
        context, ad = decision["context"], ads[str(imp["candidate_id"])]  # type: ignore[index]
        if not isinstance(context, dict) or not isinstance(ad, dict):
            dropped += 1
            continue
        hour = pl.Series([str(imp["hour"])]).str.to_datetime().dt.strftime("%y%m%d%H")[0]
        rows.append(
            {
                **{k: context[k] for k in IMPRESSION_COLUMNS if k in context},
                **{k: ad[k] for k in AD_FIELDS},
                "id": imp["impression_id"],
                "hour": hour,
                "click": int(str(imp["impression_id"]) in clicked),
            }
        )
    frame = pl.DataFrame(rows, schema=IMPRESSION_COLUMNS) if rows else pl.DataFrame(schema=IMPRESSION_COLUMNS)
    return frame.select(list(IMPRESSION_COLUMNS)), dropped


def main(log_path: str, out: str) -> None:
    """Write the training CSV and report what was dropped."""
    rows, dropped = build_rows(Path(log_path).read_text().splitlines())
    rows.write_csv(out)
    print(f"{rows.height:,} rows, {int(rows['click'].sum()):,} clicks, {dropped:,} impressions without a decision")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
