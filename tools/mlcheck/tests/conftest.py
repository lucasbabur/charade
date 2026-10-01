"""A synthetic project that passes every check; tests break one thing at a time."""

import hashlib
import json
import textwrap
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

import mlcheck.checks  # noqa: F401
from mlcheck.config import load_config
from mlcheck.context import Context
from mlcheck.registry import CHECKS, run_check
from mlcheck.result import CheckResult

START = datetime(2014, 10, 21)
HOURS = 24 * 6
ROWS_PER_HOUR = 400
SPLIT_DAYS = {"train": (0, 4), "val": (4, 5), "test": (5, 6)}

CONFIG = """
[tool.mlcheck]
package = "demo_pkg"
source_root = "src"
serving_package = "demo_pkg.serving"
features_package = "demo_pkg.features"
training_only_modules = ["torch", "lightgbm"]
artifacts_dir = "artifacts/current"
primary_model = "dcn"
baseline_model = "prior"
required_slices = ["genre", "day"]

[tool.mlcheck.thresholds]
min_ess = 100

[tool.mlcheck.data]
path = "events.csv"
id = "id"
label = "click"
time = "hour"
time_format = "%y%m%d%H"
categorical = ["site", "character_id"]

[[tool.mlcheck.data.foreign_keys]]
column = "character_id"
ref_path = "characters.csv"
ref_column = "character_id"
ref_created_at = "created_at"
"""

SOURCES = {
    "__init__.py": "",
    "features/__init__.py": "import numpy as np\n\n\ndef build(x):\n    return np.asarray(x)\n",
    "serving/__init__.py": "",
    "serving/app.py": "from demo_pkg.features import build\n\n\ndef score(x):\n    return build(x)\n",
    "training/__init__.py": "",
    "training/train.py": textwrap.dedent(
        """
        import numpy as np
        import torch

        from demo_pkg.features import build


        def fit(train_x, val_preds, val_y, calibrator, seed):
            rng = np.random.default_rng(seed)
            torch.manual_seed(seed)
            calibrator.fit(val_preds, val_y)  # mlcheck: ignore[MLS007]
            return build(train_x), rng.random()
        """
    ),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, default=str))


def _events(rng: np.random.Generator) -> pl.DataFrame:
    n = HOURS * ROWS_PER_HOUR
    ts = [START + timedelta(hours=h) for h in np.repeat(np.arange(HOURS), ROWS_PER_HOUR).tolist()]
    x = rng.normal(size=n)
    truth = 1 / (1 + np.exp(-(-1.5 + 0.9 * x)))
    return pl.DataFrame(
        {
            "id": [f"e{i}" for i in range(n)],
            "ts": ts,
            "hour": [t.strftime("%y%m%d%H") for t in ts],
            "click": rng.binomial(1, truth).astype(np.int64),
            "truth": truth,
            "site": rng.choice(["s1", "s2", "s3"], size=n),
            "character_id": rng.choice(["c1", "c2", "c3", "c4"], size=n),
            "genre": rng.choice(["romance", "horror", "mentor"], size=n),
        }
    )


def _split_of(ts: datetime) -> str:
    day = (ts - START).days
    return next(name for name, (lo, hi) in SPLIT_DAYS.items() if lo <= day < hi)


def _windows() -> dict[str, dict[str, datetime]]:
    return {
        name: {"start": START + timedelta(days=lo), "end": START + timedelta(days=hi) - timedelta(hours=1)}
        for name, (lo, hi) in SPLIT_DAYS.items()
    }


def _predictions(events: pl.DataFrame) -> pl.DataFrame:
    evaluation = events.filter(pl.col("split") != "train")
    base_rate = float(events.filter(pl.col("split") == "train")["click"].mean())  # type: ignore[arg-type]
    common = evaluation.select(
        "id",
        "split",
        pl.col("click").alias("label"),
        "ts",
        pl.col("genre").alias("slice_genre"),
        pl.col("ts").dt.date().cast(pl.Utf8).alias("slice_day"),
    )
    primary = common.with_columns(pl.lit("dcn").alias("model"), evaluation["truth"].alias("pred"))
    baseline = common.with_columns(pl.lit("prior").alias("model"), pl.lit(base_rate).alias("pred"))
    return pl.concat([primary, baseline]).select(
        "id", "split", "model", "label", "pred", "ts", "slice_genre", "slice_day"
    )


def _write_sources(root: Path) -> None:
    for relative, body in SOURCES.items():
        path = root / "src" / "demo_pkg" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)


def _write_reports(art: Path) -> None:
    _write_json(
        art / "leakage.json",
        {"shuffled_label_auc": 0.501, "feature_auc": {"site": 0.52, "x": 0.71}, "adversarial_auc": 0.55},
    )
    _write_json(
        art / "parity.json",
        {"n_rows": 1000, "train_serve_max_abs_diff": 0.0, "categorical_mismatches": 0, "onnx_max_abs_diff": 2e-7},
    )
    _write_json(
        art / "latency.json",
        {
            "source": "locust",
            "n_candidates": 100,
            "requests": 24000,
            "duration_s": 120,
            "p50_ms": 7.9,
            "p95_ms": 14.2,
            "p99_ms": 21.5,
            "error_rate": 0.0,
        },
    )
    _write_json(
        art / "ope.json",
        {
            "policies": [
                {
                    "name": "greedy",
                    "estimator": "snips",
                    "value": 0.19,
                    "ci_low": 0.18,
                    "ci_high": 0.2,
                    "ess": 5000,
                    "n": 9600,
                    "max_weight": 8.0,
                }
            ]
        },
    )
    _write_json(
        art / "drift.json", {"reference": "train", "psi": {"site": {"2014-10-26": 0.03}, "genre": {"2014-10-26": 0.01}}}
    )
    decisions = [
        {
            "request_id": "r1",
            "candidates": [
                {"candidate_id": "a", "pctr": 0.2, "gated": False},
                {"candidate_id": "b", "pctr": 0.4, "gated": True, "gate_reasons": ["brand_safety"]},
            ],
            "chosen_id": "a",
            "propensity": 1.0,
            "explored": False,
        },
        {
            "request_id": "r2",
            "candidates": [{"candidate_id": "c", "pctr": 0.1, "gated": True, "gate_reasons": ["frequency_cap"]}],
            "chosen_id": None,
            "propensity": None,
            "explored": False,
        },
        {
            "request_id": "r3",
            "candidates": [
                {"candidate_id": "d", "pctr": 0.1, "gated": False},
                {"candidate_id": "e", "pctr": 0.3, "gated": False},
            ],
            "chosen_id": "d",
            "propensity": 0.02,
            "explored": True,
        },
    ]
    (art / "decisions.jsonl").write_text("\n".join(json.dumps(d) for d in decisions))


def build_project(root: Path) -> None:
    """Write sources, raw data and a complete, passing run into `root`."""
    rng = np.random.default_rng(7)
    (root / "pyproject.toml").write_text(CONFIG)
    _write_sources(root)
    events = _events(rng)
    events = events.with_columns(pl.col("ts").map_elements(_split_of, return_dtype=pl.Utf8).alias("split"))
    events.select("id", "hour", "click", "site", "character_id").write_csv(root / "events.csv")
    pl.DataFrame({"character_id": ["c1", "c2", "c3", "c4"], "created_at": ["2014-01-01"] * 4}).write_csv(
        root / "characters.csv"
    )
    art = root / "artifacts" / "current"
    art.mkdir(parents=True)
    events.select("id", "split", "ts").write_parquet(art / "splits.parquet")
    _predictions(events).write_parquet(art / "predictions.parquet")
    windows = _windows()
    _write_json(
        art / "manifest.json",
        {
            "run_id": "run-1",
            "created_at": START + timedelta(days=7),
            "git_sha": "a" * 40,
            "git_dirty": False,
            "config_hash": "c0ffee00",
            "data_sha256": {"events.csv": _sha(root / "events.csv"), "characters.csv": _sha(root / "characters.csv")},
            "windows": windows,
            "fitted_artifacts": [
                {"name": "vocab", "fit_split": "train", "fit_end": windows["train"]["end"]},
                {"name": "calibrator", "fit_split": "val", "fit_end": windows["val"]["end"]},
            ],
            "model_seeds": {"dcn": [0, 1, 2]},
            "library_versions": {"torch": "2.8.0"},
        },
    )
    (art / "evaluation_ledger.jsonl").write_text(
        json.dumps(
            {
                "timestamp": str(START + timedelta(days=7)),
                "run_id": "run-1",
                "model_version": "dcn@run-1",
                "split": "test",
            }
        )
    )
    _write_reports(art)


@pytest.fixture(scope="session")
def golden(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("golden")
    build_project(root)
    return root


@pytest.fixture
def project(tmp_path: Path) -> Path:
    build_project(tmp_path)
    return tmp_path


type Runner = Callable[[Path, str], CheckResult]


@pytest.fixture
def run() -> Runner:
    def _run(root: Path, code: str) -> CheckResult:
        return run_check(CHECKS[code], Context(root, load_config(root)))

    return _run


def edit_json(path: Path, change: Callable[[dict[str, Any]], None]) -> None:
    payload = json.loads(path.read_text())
    change(payload)
    path.write_text(json.dumps(payload))
