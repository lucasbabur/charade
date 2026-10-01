"""Sample rankings for two chat contexts (`uv run poe samples`) -> reports/sample_rankings.json.

1. A mature romance character mid-session, returning user, on the app surface.
2. A brand-new sfw mentor character (not in the character table) on its first turn, new user.
Both get the same eight candidates, including one from advertiser 48 (sfw-only, illustrative),
so the brand-safety gate and the genre-dependent ordering are visible side by side.
"""

import json
from datetime import datetime
from pathlib import Path

import polars as pl
from fastapi.testclient import TestClient

from charade.analysis.requests import creative_pool, request_from_row
from charade.config import get_settings
from charade.models.dataset import build_frame
from charade.serving.app import create_app
from charade.serving.store import FeatureStore, Served

OUT = Path("reports/sample_rankings.json")


def run(out: Path = OUT) -> list[dict[str, object]]:
    """Call the real API in-process and save request + response pairs."""
    settings = get_settings()
    frame = build_frame(settings, settings.data_dir)
    test = frame.filter(pl.col("split") == "test")
    pool = creative_pool(frame.filter(pl.col("split") == "train"))
    candidates = pl.concat(
        [pool.filter(pl.col("C21") != "48").head(7), pool.filter(pl.col("C21") == "48").head(1)]
    ).to_dicts()
    romance = test.filter(
        (pl.col("genre") == "romance")
        & (pl.col("safety_tier") == "mature")
        & (pl.col("user_imps") > 5)
        & (pl.col("conversation_turn") >= 6)
    ).row(0, named=True)
    fresh = test.filter((pl.col("user_imps") == 0) & (pl.col("conversation_turn") == 1)).row(0, named=True)
    first = request_from_row(romance, candidates, "sample-romance-mature-returning")
    second = request_from_row(fresh, candidates, "sample-new-mentor-character").model_copy(
        update={"character_id": "brand-new-character"}
    )
    second_body = json.loads(second.model_dump_json())
    second_body["character"] = {
        "genre": "mentor",
        "safety_tier": "sfw",
        "creator_type": "community",
        "num_interactions": 0,
        "created_at": datetime(2014, 10, 29).isoformat(),
    }
    samples: list[dict[str, object]] = []
    with TestClient(create_app()) as client:
        store = client.app.state.runtime.store  # pyright: ignore[reportAttributeAccessIssue, reportFunctionMemberAccess]
        _replay_history(store, frame, romance)
        for label, body in (
            ("mature romance, returning user, turn 6+", json.loads(first.model_dump_json())),
            ("new sfw mentor character, new user, turn 1", second_body),
        ):
            response = client.post("/v1/rank", json=body)
            response.raise_for_status()
            samples.append({"context": label, "request": body, "response": response.json()})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(samples, indent=2, default=str))
    return samples


def _replay_history(store: FeatureStore, frame: pl.DataFrame, row: dict[str, object]) -> None:
    import asyncio  # noqa: PLC0415

    from charade.serving.assemble import epoch_hour  # noqa: PLC0415

    events = frame.filter((pl.col("user") == row["user"]) & (pl.col("ts") < pl.lit(row["ts"]))).sort("ts", "id")

    async def replay() -> None:
        for e in events.iter_rows(named=True):
            served = Served(user=e["user"], hour=epoch_hour(e["ts"]), candidate_id=e["id"], campaign=e["C17"])
            await store.record_decision(e["id"], served)
            await store.record_impression(e["id"], e["id"])
            if e["click"]:
                await store.record_click(e["id"])

    asyncio.run(replay())


if __name__ == "__main__":
    for sample in run():
        response = sample["response"]
        if not isinstance(response, dict):
            raise TypeError(response)
        print(
            sample["context"],
            response["chosen_id"],
            [(r["candidate_id"], round(r["pctr"], 4), r["gate_reasons"]) for r in response["ranked"]],
        )
