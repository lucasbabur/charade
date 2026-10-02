"""Load-test inputs and summary (`uv run poe loadtest`, see docs/07-serving-operations.md).

`build_requests` writes realistic `/v1/rank` bodies: test-day contexts with N candidates drawn from
the creative pool, weighted by volume. `summarize` turns Locust's CSV stats into `latency.json`.
"""

import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import polars as pl

from charade.analysis.requests import creative_pool, request_from_row
from charade.config import get_settings
from charade.models.dataset import build_frame

OUT = Path("artifacts/loadtest")
N_REQUESTS = 2000
N_CANDIDATES = 100


def build_requests(out: Path = OUT, n_requests: int = N_REQUESTS, n_candidates: int = N_CANDIDATES) -> Path:
    """Write `requests.jsonl` (one RankRequest per line)."""
    settings = get_settings()
    frame = build_frame(settings, settings.data_dir)
    test = frame.filter(pl.col("split") == "test").sample(n_requests, seed=settings.seed)
    pool = creative_pool(frame.filter(pl.col("split") != "test")).head(1000)
    weights = pool["volume"].to_numpy().astype(np.float64)
    rng = np.random.default_rng(settings.seed)
    creatives = pool.to_dicts()
    out.mkdir(parents=True, exist_ok=True)
    path = out / "requests.jsonl"
    with path.open("w") as handle:
        for i, row in enumerate(test.iter_rows(named=True)):
            picks = rng.choice(len(creatives), size=n_candidates, replace=False, p=weights / weights.sum())
            request = request_from_row(row, [creatives[j] for j in picks], f"load-{i}")
            handle.write(request.model_dump_json() + "\n")
    return path


def summarize(stats_csv: Path, host: str, n_candidates: int = N_CANDIDATES) -> dict[str, object]:
    """Locust `*_stats.csv` -> latency.json fields for the /v1/rank row, naming the bundle `host` served."""
    with urllib.request.urlopen(f"{host.rstrip('/')}/v1/model", timeout=5) as response:  # noqa: S310 - local host
        served = json.loads(response.read())["bundle_sha256"]
    table = pl.read_csv(stats_csv).filter(pl.col("Name") != "Aggregated")
    stats = table.filter(pl.col("Name") == "/v1/rank").row(0, named=True)
    requests = int(stats["Request Count"])
    return {
        "source": "locust",
        "n_candidates": n_candidates,
        "requests": requests,
        "duration_s": requests / float(stats["Requests/s"]),
        "p50_ms": float(stats["50%"]),
        "p95_ms": float(stats["95%"]),
        "p99_ms": float(stats["99%"]),
        # Errors over every endpoint (ranking and the impression/click events), latency for ranking only.
        "error_rate": int(table["Failure Count"].sum()) / max(int(table["Request Count"].sum()), 1),
        "bundle_sha256": served,
    }


if __name__ == "__main__":
    if sys.argv[1:2] == ["summarize"]:
        summary = summarize(Path(sys.argv[2]), sys.argv[3])
        get_settings().artifacts_dir.joinpath("latency.json").write_text(json.dumps(summary, indent=2))
        Path("reports/serving").mkdir(parents=True, exist_ok=True)
        Path("reports/serving/latency.json").write_text(json.dumps(summary, indent=2))
        print(summary)
    else:
        print(build_requests())
