"""Turn serving logs into training rows (`python -m charade.data.events <log.jsonl> <out.csv> [as_of]`).

The API writes three JSON events (CloudWatch -> Firehose -> S3 in production; Firehose extracts the
log messages, so S3 holds gzipped JSON lines):
    decision    request_id, hour, character_id, chosen_id, context (request fields), ads (per-candidate attributes)
    impression  impression_id, request_id, candidate_id, hour   (taken from the stored decision)
    click       impression_id                                   (may arrive up to CLICK_WINDOW_HOURS later)

Each impression joins its decision on request_id; its ad is the decision's chosen candidate, and
click = 1 if a click event exists for the impression. Delivery is at least once (the API re-emits on
retried writes), so identical repeats are deduplicated. Nothing is guessed; a row is dropped and
counted by reason when:
    no decision            the impression's decision is missing (log loss, retention)
    conflicting decision   one request id logged with two different decisions
    conflicting impression one impression id logged for two different requests or candidates
    not the served ad      the impression names a candidate its decision did not choose
    label pending          a click for the impression may still arrive at `as_of` (its hour plus one, since
                           the hour is truncated, plus CLICK_WINDOW_HOURS), so click = 0 would be a guess
The result has exactly the columns of `impressions.csv`, so the training pipeline consumes it unchanged.
"""

import gzip
import json
import sys
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from charade.data.load import IMPRESSION_COLUMNS

AD_FIELDS = ("banner_pos", "C14", "C15", "C16", "C17", "C18", "C19", "C21")
CLICK_WINDOW_HOURS = 1
"""Clicks are attributed for 1 h after an impression (the store then forgets it), so labels mature then.
Clicks on an ad in a chat come within minutes; the label maturity, not the click delay, is what training waits on."""
DECISION_IDENTITY = ("request_id", "hour", "character_id", "chosen_id", "context", "ads")
IMPRESSION_IDENTITY = ("impression_id", "request_id", "candidate_id", "hour")


def _records(lines: Iterable[str]) -> Iterable[dict[str, object]]:
    for raw in lines:
        line = raw.strip()
        if line.startswith("{"):
            record = json.loads(line)
            if record.get("event") in {"decision", "impression", "click"}:
                yield record


def _first_unless_conflicting(
    records: Iterable[dict[str, object]], key: str, identity: tuple[str, ...]
) -> tuple[dict[str, dict[str, object]], set[str]]:
    """First record per key, and the keys logged with two different identities."""
    first: dict[str, dict[str, object]] = {}
    conflicting: set[str] = set()
    for record in records:
        kept = first.setdefault(str(record[key]), record)
        if any(kept.get(f) != record.get(f) for f in identity):
            conflicting.add(str(record[key]))
    return first, conflicting


def _naive_utc(value: object) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    return parsed.astimezone(UTC).replace(tzinfo=None) if parsed.tzinfo else parsed


def build_rows(lines: Iterable[str], as_of: datetime) -> tuple[pl.DataFrame, Counter[str]]:
    """(training rows in the impressions.csv schema, dropped impressions by reason) as of a UTC time."""
    records = list(_records(lines))
    decisions, bad_decisions = _first_unless_conflicting(
        (r for r in records if r["event"] == "decision"), "request_id", DECISION_IDENTITY
    )
    impressions, bad_impressions = _first_unless_conflicting(
        (r for r in records if r["event"] == "impression"), "impression_id", IMPRESSION_IDENTITY
    )
    clicked = {str(r["impression_id"]) for r in records if r["event"] == "click"}
    # `hour` is truncated, so an impression logged at hour h happened as late as h + 1.
    mature_before = _naive_utc(as_of.isoformat()) - timedelta(hours=CLICK_WINDOW_HOURS + 1)
    rows: list[dict[str, object]] = []
    dropped: Counter[str] = Counter()
    for impression_id, imp in impressions.items():
        request_id = str(imp["request_id"])
        decision = decisions.get(request_id)
        reason = (
            "conflicting impression"
            if impression_id in bad_impressions
            else "no decision"
            if decision is None
            else "conflicting decision"
            if request_id in bad_decisions
            else "not the served ad"
            if decision.get("chosen_id") != imp["candidate_id"]
            else "label pending"
            if _naive_utc(imp["hour"]) > mature_before
            else None
        )
        if reason is not None:
            dropped[reason] += 1
            continue
        assert decision is not None  # noqa: S101 - "no decision" handled above
        context, ads = decision["context"], decision["ads"]
        ad = ads[str(imp["candidate_id"])]  # type: ignore[index]
        rows.append(
            {
                **{k: context[k] for k in IMPRESSION_COLUMNS if k in context},  # type: ignore[operator, index]
                **{k: ad[k] for k in AD_FIELDS},
                "id": impression_id,
                "hour": _naive_utc(imp["hour"]).strftime("%y%m%d%H"),
                "click": int(impression_id in clicked),
            }
        )
    frame = pl.DataFrame(rows, schema=IMPRESSION_COLUMNS) if rows else pl.DataFrame(schema=IMPRESSION_COLUMNS)
    return frame.select(list(IMPRESSION_COLUMNS)), dropped


def main(log_path: str, out: str, as_of: str | None = None) -> None:
    """Write the training CSV and report what was dropped (as of now, UTC, unless given). Reads .gz too."""
    when = datetime.fromisoformat(as_of) if as_of else datetime.now(UTC)
    path = Path(log_path)
    text = gzip.decompress(path.read_bytes()).decode() if path.suffix == ".gz" else path.read_text()
    rows, dropped = build_rows(text.splitlines(), when)
    rows.write_csv(out)
    reasons = ", ".join(f"{n:,} {reason}" for reason, n in sorted(dropped.items())) or "none"
    print(f"{rows.height:,} rows, {int(rows['click'].sum()):,} clicks; dropped: {reasons}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
