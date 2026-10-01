import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from charade.config import Settings
from charade.data.events import build_rows
from charade.data.load import IMPRESSION_COLUMNS, load_joined
from charade.serving.app import create_app
from charade.serving.runtime import load_runtime
from charade.serving.store import MemoryStore
from tests.conftest import FIXTURES


def test_serving_logs_rebuild_training_rows(
    bundle: Settings,
    sample_body: dict[str, object],
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Rank -> impression -> late click through the API; the logs become impressions.csv rows."""
    device = {k: sample_body[k] for k in ("device_id", "device_ip", "device_model")}
    capsys.readouterr()
    with TestClient(create_app(load_runtime(bundle, MemoryStore()))) as client:
        served = []
        for i in range(3):
            body = sample_body | {"request_id": f"req-{i}"}
            chosen = client.post("/v1/rank", json=body).json()["chosen_id"]
            event = device | {
                "impression_id": f"imp-{i}",
                "request_id": f"req-{i}",
                "candidate_id": chosen,
                "hour": sample_body["hour"],
                "campaign_id": "c",
            }
            assert client.post("/v1/events/impression", json=event).json()["outcome"] == "recorded"
            client.post("/v1/events/impression", json=event)  # retry: must not duplicate the row
            served.append(chosen)
        client.post("/v1/events/click", json={"impression_id": "imp-1"})
        client.post(
            "/v1/events/impression",
            json=device
            | {
                "impression_id": "orphan",
                "request_id": "never-ranked",
                "candidate_id": "x",
                "hour": sample_body["hour"],
                "campaign_id": "c",
            },
        )
    rows, dropped = build_rows(capsys.readouterr().out.splitlines())
    assert dropped == 1
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

    device = {k: sample_body[k] for k in ("device_id", "device_ip", "device_model")}
    capsys.readouterr()
    with TestClient(create_app(load_runtime(bundle, MemoryStore())), raise_server_exceptions=False) as client:
        chosen = client.post("/v1/rank", json=sample_body | {"request_id": "req-x"}).json()["chosen_id"]
        event = device | {
            "impression_id": "imp-x",
            "request_id": "req-x",
            "candidate_id": chosen,
            "hour": sample_body["hour"],
            "campaign_id": "c",
        }
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
    rows, _ = build_rows(capsys.readouterr().out.splitlines())
    assert rows["id"].to_list() == ["imp-x"]
    assert rows["click"].to_list() == [1]
