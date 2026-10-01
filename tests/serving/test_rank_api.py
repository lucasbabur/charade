import copy
from pathlib import Path

import polars as pl
import pytest
from fastapi.testclient import TestClient

from charade.analysis.requests import creative_pool, request_from_row
from charade.config import Settings
from charade.features.counters import UserHistory
from charade.models.dataset import build_frame
from charade.serving.app import create_app
from charade.serving.runtime import Runtime, load_runtime
from charade.serving.store import Impression, MemoryStore, Outcome, StoreUnavailableError
from tests.conftest import FIXTURES


class BrokenStore(MemoryStore):
    async def get(self, user: str) -> UserHistory | None:
        raise StoreUnavailableError(user)

    async def record_impression(self, impression_id: str, impression: Impression) -> Outcome:
        raise StoreUnavailableError(impression_id)


@pytest.fixture(scope="module")
def sample_body(bundle: Settings) -> dict[str, object]:
    frame = build_frame(FIXTURES, None)
    test = frame.filter(pl.col("split") == "test")
    pool = creative_pool(frame).head(6).to_dicts()
    row = test.row(0, named=True)
    return request_from_row(row, pool, "req-1").model_dump(mode="json")


def _client(bundle: Settings, store: MemoryStore | None = None) -> TestClient:
    return TestClient(create_app(load_runtime(bundle, store or MemoryStore())))


def test_rank_returns_every_candidate_with_a_decision(bundle: Settings, sample_body: dict[str, object]) -> None:
    with _client(bundle) as client:
        response = client.post("/v1/rank", json=sample_body)
    assert response.status_code == 200
    body = response.json()
    assert len(body["ranked"]) == 6
    assert body["chosen_id"] in {r["candidate_id"] for r in body["ranked"] if not r["gated"]}
    assert 0 < body["propensity"] <= 1
    assert all(r["pctr_low"] <= r["pctr"] <= r["pctr_high"] for r in body["ranked"])


def test_unknown_character_uses_request_metadata_or_strictest_defaults(
    bundle: Settings, sample_body: dict[str, object]
) -> None:
    body = copy.deepcopy(sample_body) | {"character_id": "never-seen"}
    with _client(bundle) as client:
        default = client.post("/v1/rank", json=body).json()
        provided = client.post(
            "/v1/rank",
            json=body | {"character": {"genre": "romance", "safety_tier": "sfw", "created_at": "2014-10-29T00:00:00"}},
        ).json()
    assert default["cold_start"]["character"] is True
    assert default["ranked"] != provided["ranked"]


def test_store_outage_degrades_instead_of_failing(bundle: Settings, sample_body: dict[str, object]) -> None:
    with _client(bundle, BrokenStore()) as client:
        response = client.post("/v1/rank", json=sample_body)
        event = client.post(
            "/v1/events/impression",
            json={
                "impression_id": "i",
                "hour": "14102912",
                "device_id": "x",
                "device_ip": "y",
                "device_model": "z",
                "campaign_id": "c",
            },
        )
    assert response.status_code == 200
    assert response.json()["degraded"] is True
    assert event.status_code == 503


def test_impression_and_click_events_feed_the_next_request(bundle: Settings, sample_body: dict[str, object]) -> None:
    store = MemoryStore()
    event = {k: sample_body[k] for k in ("device_id", "device_ip", "device_model")} | {
        "impression_id": "imp-1",
        "hour": "14102000",
        "campaign_id": "c",
    }
    with _client(bundle, store) as client:
        assert client.post("/v1/rank", json=sample_body).json()["cold_start"]["user"] is True
        assert client.post("/v1/events/impression", json=event).json() == {"outcome": "recorded"}
        assert client.post("/v1/events/impression", json=event).json() == {"outcome": "duplicate"}
        assert client.post("/v1/events/click", json={"impression_id": "imp-1"}).json() == {"outcome": "recorded"}
        assert client.post("/v1/events/click", json={"impression_id": "imp-1"}).json() == {"outcome": "duplicate"}
        assert client.post("/v1/events/click", json={"impression_id": "unknown"}).status_code == 404
        assert client.post("/v1/rank", json=sample_body).json()["cold_start"]["user"] is False
    history = next(iter(store.users.values()))
    assert (history.imps, history.clicks) == (1, 1)


def test_duplicates_and_inconsistent_turns_are_repaired_with_warnings(
    bundle: Settings, sample_body: dict[str, object]
) -> None:
    body = copy.deepcopy(sample_body)
    candidates = body["candidates"]
    assert isinstance(candidates, list)
    body |= {"candidates": [*candidates, candidates[0]], "conversation_turn": 50, "session_msg_count": 3}
    with _client(bundle) as client:
        response = client.post("/v1/rank", json=body).json()
    assert len(response["ranked"]) == 6
    assert len(response["warnings"]) == 2


@pytest.mark.parametrize(
    "change",
    [
        {"candidates": []},
        {"conversation_turn": 0},
        {"hour": "not-a-time"},
        {"unexpected": 1},
    ],
)
def test_invalid_requests_are_rejected(
    bundle: Settings, sample_body: dict[str, object], change: dict[str, object]
) -> None:
    with _client(bundle) as client:
        assert client.post("/v1/rank", json=sample_body | change).status_code == 422


def test_compact_hour_format_is_accepted(bundle: Settings, sample_body: dict[str, object]) -> None:
    with _client(bundle) as client:
        assert client.post("/v1/rank", json=sample_body | {"hour": "14102912"}).status_code == 200


def test_not_ready_without_a_bundle(tmp_path: Path, bundle: Settings) -> None:
    empty = bundle.model_copy(update={"artifacts_dir": tmp_path})
    runtime: Runtime = load_runtime(empty, MemoryStore())
    with TestClient(create_app(runtime)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        assert client.post("/v1/rank", json={}).status_code in {422, 503}


def test_ops_endpoints(bundle: Settings) -> None:
    with _client(bundle) as client:
        assert client.get("/ready").status_code == 200
        assert client.get("/v1/model").json()["calibrator"]
        assert b"charade_request_seconds" in client.get("/metrics").content


def test_decisions_are_logged_as_json_events(
    bundle: Settings, sample_body: dict[str, object], capsys: pytest.CaptureFixture[str]
) -> None:
    import json  # noqa: PLC0415

    with _client(bundle) as client:
        client.post("/v1/rank", json=sample_body)
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    decision = next(r for r in records if r["event"] == "decision")
    assert decision["request_id"] == "req-1"
    assert "propensity" in decision
    assert len(decision["candidates"]) == 6
    assert {"candidate_id", "pctr", "value", "gate_reasons"} <= set(decision["candidates"][0])
