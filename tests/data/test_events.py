import gzip
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from charade.config import Settings
from charade.data.events import build_rows, main
from charade.data.load import IMPRESSION_COLUMNS, load_joined
from charade.serving.app import create_app
from charade.serving.runtime import load_runtime
from charade.serving.store import MemoryStore
from tests.conftest import FIXTURES

MATURE = datetime(2015, 1, 1)
"""Long after every fixture impression: all labels are final."""


def test_serving_logs_rebuild_training_rows(
    bundle: Settings,
    sample_body: dict[str, object],
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Rank -> impression -> late click through the API; the logs become impressions.csv rows."""
    capsys.readouterr()
    with TestClient(create_app(load_runtime(bundle, MemoryStore()))) as client:
        served = []
        for i in range(3):
            body = sample_body | {"request_id": f"req-{i}"}
            chosen = client.post("/v1/rank", json=body).json()["chosen_id"]
            event = {"impression_id": f"imp-{i}", "request_id": f"req-{i}"}
            assert client.post("/v1/events/impression", json=event).json()["outcome"] == "recorded"
            client.post("/v1/events/impression", json=event)  # retry: must not duplicate the row
            served.append(chosen)
        client.post("/v1/events/click", json={"impression_id": "imp-1"})
        assert (
            client.post("/v1/events/impression", json={"impression_id": "orphan", "request_id": "x"}).status_code == 404
        )
    lines = capsys.readouterr().out.splitlines()
    rows, dropped = build_rows(lines, MATURE)
    assert dropped == {}
    pending, held = build_rows(lines, datetime.fromisoformat(str(sample_body["hour"])) + timedelta(hours=1))
    assert (pending.height, held) == (0, {"label pending": 3})
    assert rows.columns == list(IMPRESSION_COLUMNS)
    assert rows["id"].to_list() == ["imp-0", "imp-1", "imp-2"]
    assert rows["click"].to_list() == [0, 1, 0]
    candidates = {c["candidate_id"]: c for c in sample_body["candidates"]}  # type: ignore[union-attr]
    assert rows["C14"].to_list() == [candidates[c]["C14"] for c in served]
    assert rows["character_id"][0] == sample_body["character_id"]
    # The rebuilt rows load through the same contract-checked loader training uses.
    rows.write_csv(tmp_path / "impressions.csv")
    shutil.copy(FIXTURES / "characters.csv", tmp_path / "characters.csv")
    assert load_joined(tmp_path).height == 3


def test_logging_failure_after_state_write_is_repaired_by_the_retry(
    bundle: Settings,
    sample_body: dict[str, object],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review reproduction: the state write succeeds, the event log fails, the client retries."""
    from charade.serving import app as app_module  # noqa: PLC0415

    capsys.readouterr()
    with TestClient(create_app(load_runtime(bundle, MemoryStore())), raise_server_exceptions=False) as client:
        client.post("/v1/rank", json=sample_body | {"request_id": "req-x"})
        event = {"impression_id": "imp-x", "request_id": "req-x"}
        real_info = app_module.log.info

        def failing(name: str, **kw: object) -> None:
            if name in {"impression", "click"}:
                raise RuntimeError("log sink down")
            real_info(name, **kw)

        monkeypatch.setattr(app_module.log, "info", failing)
        assert client.post("/v1/events/impression", json=event).status_code == 500
        assert client.post("/v1/events/click", json={"impression_id": "imp-x"}).status_code == 500
        monkeypatch.setattr(app_module.log, "info", real_info)
        assert client.post("/v1/events/impression", json=event).json()["outcome"] == "duplicate"
        assert client.post("/v1/events/click", json={"impression_id": "imp-x"}).json()["outcome"] == "duplicate"
        client.post("/v1/events/impression", json=event)  # a third delivery must not duplicate the row
    rows, _ = build_rows(capsys.readouterr().out.splitlines(), MATURE)
    assert rows["id"].to_list() == ["imp-x"]
    assert rows["click"].to_list() == [1]


def _line(event: str, **fields: object) -> str:
    return json.dumps({"event": event, **fields})


def test_conflicting_and_unserved_events_are_dropped_by_reason() -> None:
    """Review reproduction: a later decision under the same request id must not rewrite a served row."""
    context = {"site_id": "s", "device_id": "d"}
    ads = {"a": dict.fromkeys(("banner_pos", "C14", "C15", "C16", "C17", "C18", "C19", "C21"), "1")}
    decision = {"hour": "2014-10-29T10:00:00", "character_id": "c1", "chosen_id": "a", "ads": ads}
    lines = [
        _line("decision", request_id="r1", context=context, **decision),
        _line("decision", request_id="r1", context=context, **decision),  # identical redelivery
        _line("decision", request_id="r2", context=context, **decision),
        _line("decision", request_id="r2", context=context | {"device_id": "other"}, **decision),
        _line("impression", impression_id="i1", request_id="r1", candidate_id="a", hour="2014-10-29T10:00:00"),
        _line("impression", impression_id="i2", request_id="r2", candidate_id="a", hour="2014-10-29T10:00:00"),
        _line("impression", impression_id="i3", request_id="r1", candidate_id="b", hour="2014-10-29T10:00:00"),
        _line("impression", impression_id="i4", request_id="r1", candidate_id="a", hour="2014-10-29T10:00:00"),
        _line("impression", impression_id="i4", request_id="r2", candidate_id="a", hour="2014-10-29T10:00:00"),
        _line("click", impression_id="i1"),
    ]
    rows, dropped = build_rows(lines, MATURE)
    assert rows["id"].to_list() == ["i1"]
    assert rows["click"].to_list() == [1]
    assert dropped == {"conflicting decision": 1, "not the served ad": 1, "conflicting impression": 1}


def test_gzipped_json_lines_are_read(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """What Firehose writes to S3 after message extraction: gzipped JSON lines."""
    ads = {"a": dict.fromkeys(("banner_pos", "C14", "C15", "C16", "C17", "C18", "C19", "C21"), "1")}
    hour = "2014-10-29T10:00:00"
    lines = [
        _line(
            "decision", request_id="r", hour=hour, character_id="c", chosen_id="a", context={"site_id": "s"}, ads=ads
        ),
        _line("impression", impression_id="i", request_id="r", candidate_id="a", hour=hour),
    ]
    path = tmp_path / "part.gz"
    path.write_bytes(gzip.compress("\n".join(lines).encode()))
    main(str(path), str(tmp_path / "rows.csv"), "2015-01-01T00:00:00")
    assert capsys.readouterr().out.startswith("1 rows, 0 clicks; dropped: none")
