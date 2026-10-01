"""Train/serve parity (`uv run poe parity`): the API path must build the same model inputs as training.

For a sample of test impressions, the online store is rebuilt by replaying every earlier impression
of those users through the store, in time order, exactly as `POST /v1/events/impression` would.
Each sampled impression is then turned into an API request with its logged ad as the only
candidate and assembled by the serving code. The encoded arrays are compared with the offline
pipeline's encoding of the same rows. Writes `parity.json` (with the ONNX check from training).
"""

import asyncio
import json
from pathlib import Path

import numpy as np
import polars as pl

from charade.analysis.requests import request_from_row
from charade.config import Settings, get_settings
from charade.features.spec import encode
from charade.models.dataset import build_frame
from charade.scoring.scorer import Scorer
from charade.serving.assemble import AD_FIELDS, assemble, epoch_hour
from charade.serving.store import MemoryStore

SAMPLE = 2000


async def _compare(
    frame: pl.DataFrame, sample_ids: set[str], scorer: Scorer, characters: dict[str, dict[str, object]]
) -> tuple[float, int, int]:
    store = MemoryStore()
    worst, mismatches, checked = 0.0, 0, 0
    for row in frame.sort("ts", "id").iter_rows(named=True):
        if row["id"] in sample_ids:
            request = request_from_row(row, [{f: row[f] for f in AD_FIELDS}], row["id"])
            history = await store.get(row["user"])
            _, online = assemble(request, characters[row["character_id"]], history, scorer.spec)
            offline = encode(scorer.spec, pl.DataFrame([row]))
            worst = max(worst, float(np.abs(online.dense - offline.dense).max(initial=0.0)))
            mismatches += int((online.categorical != offline.categorical).any())
            checked += 1
        await store.record(row["user"], epoch_hour(row["ts"]), row["C17"], bool(row["click"]))
    return worst, mismatches, checked


def run(settings: Settings | None = None, data_dir: Path | None = None) -> dict[str, float]:
    """Write `parity.json` into the artifacts directory."""
    settings = settings or get_settings()
    art = settings.artifacts_dir
    scorer = Scorer(art)
    frame = build_frame(data_dir or settings.data_dir, None)
    test = frame.filter(pl.col("split") == "test")
    sample = test.sample(min(SAMPLE, test.height), seed=settings.seed)
    users = frame.filter(pl.col("user").is_in(sample["user"].implode()))
    characters = {
        r["character_id"]: r
        for r in users.select("character_id", "genre", "safety_tier", "creator_type", "num_interactions", "created_at")
        .unique("character_id")
        .iter_rows(named=True)
    }
    worst, mismatches, checked = asyncio.run(_compare(users, set(sample["id"].to_list()), scorer, characters))
    export = json.loads((art / "export.json").read_text())
    report = {
        "n_rows": checked,
        "train_serve_max_abs_diff": worst,
        "categorical_mismatches": mismatches,
        "onnx_max_abs_diff": export["onnx_max_abs_diff"],
    }
    (art / "parity.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    print(run())
