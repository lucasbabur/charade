"""Every check passes on the golden project and fails on the one defect it exists to catch."""

import json
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from conftest import START, Runner, edit_json

from mlcheck.registry import CHECKS
from mlcheck.result import Status

ART = Path("artifacts/current")


@pytest.mark.parametrize("code", sorted(CHECKS))
def test_golden_project_passes(golden: Path, run: Runner, code: str) -> None:
    result = run(golden, code)
    assert result.status is Status.PASS, (result.message, result.details)


def _append_source(relative: str, body: str) -> Callable[[Path], None]:
    def mutate(root: Path) -> None:
        path = root / "src" / "demo_pkg" / relative
        path.write_text(path.read_text() + body)

    return mutate


def _json(name: str, change: Callable[[dict[str, Any]], None]) -> Callable[[Path], None]:
    return lambda root: edit_json(root / ART / name, change)


def _set(key: str, value: Any) -> Callable[[dict[str, Any]], None]:
    return lambda payload: payload.__setitem__(key, value)


def _events(change: Callable[[pl.DataFrame], pl.DataFrame]) -> Callable[[Path], None]:
    def mutate(root: Path) -> None:
        path = root / "events.csv"
        change(pl.read_csv(path, infer_schema=False)).write_csv(path)

    return mutate


def _predictions(change: Callable[[pl.DataFrame], pl.DataFrame]) -> Callable[[Path], None]:
    def mutate(root: Path) -> None:
        path = root / ART / "predictions.parquet"
        change(pl.read_parquet(path)).write_parquet(path)

    return mutate


def _primary_holdout() -> pl.Expr:
    return (pl.col("model") == "dcn") & (pl.col("split") == "test")


def _move_test_row_into_train(root: Path) -> None:
    path = root / ART / "splits.parquet"
    frame = pl.read_parquet(path)
    first_test = frame.filter(pl.col("split") == "test").row(0, named=True)
    pl.concat([frame, pl.DataFrame([{**first_test, "split": "train"}])]).write_parquet(path)


def _overlap_splits_in_time(root: Path) -> None:
    path = root / ART / "splits.parquet"
    frame = pl.read_parquet(path)
    frame.with_columns(
        pl.when(pl.col("split") == "val").then(pl.col("ts") - timedelta(days=2)).otherwise(pl.col("ts")).alias("ts")
    ).write_parquet(path)


def _vocab_fitted_on_test(manifest: dict[str, Any]) -> None:
    manifest["fitted_artifacts"][0]["fit_end"] = str(START + timedelta(days=5, hours=3))


def _serve_gated(root: Path) -> None:
    path = root / ART / "decisions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["chosen_id"] = "b"
    path.write_text("\n".join(json.dumps(r) for r in rows))


def _single_eligible_not_certain(root: Path) -> None:
    path = root / ART / "decisions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["propensity"] = 0.5
    path.write_text("\n".join(json.dumps(r) for r in rows))


def _miscalibrate(factor: float) -> Callable[[pl.DataFrame], pl.DataFrame]:
    return lambda f: f.with_columns(
        pl.when(pl.col("model") == "dcn")
        .then((pl.col("pred") * factor).clip(1e-4, 0.999))
        .otherwise(pl.col("pred"))
        .alias("pred")
    )


def _inflate_served_propensity(root: Path) -> None:
    """The bias the review found: the served ad's logged probability exceeds its true share."""
    path = root / ART / "decisions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[2]["propensity"] = 0.05
    rows[2]["candidates"][0]["propensity"] = 0.05
    path.write_text("\n".join(json.dumps(r) for r in rows))


BREAKAGES: dict[str, Callable[[Path], object]] = {
    "MLS001": _append_source("serving/app.py", "\nfrom demo_pkg.training import train\n"),
    "MLS002": lambda root: (root / "src/demo_pkg/serving/app.py").write_text("def score(x):\n    return x\n"),
    "MLS005": _append_source("training/train.py", "\nstate = torch.load('model.pt')\n"),
    "MLS006": lambda root: (root / "eda.ipynb").write_text("{}"),
    "MLD001": _events(lambda f: f.drop("site")),
    "MLD002": _events(lambda f: f.with_columns(pl.lit("2").alias("click"))),
    "MLD003": _events(lambda f: pl.concat([f, f.head(3)])),
    "MLD004": _events(
        lambda f: f.with_columns(pl.when(pl.int_range(pl.len()) < 5).then(None).otherwise(pl.col("site")).alias("site"))
    ),
    "MLD005": _events(
        lambda f: f.with_columns(
            pl.when(pl.int_range(pl.len()) == 0).then(pl.lit("1410219")).otherwise(pl.col("hour")).alias("hour")
        )
    ),
    "MLD006": _events(
        lambda f: f.filter(~pl.col("hour").str.starts_with("141026") | (pl.int_range(pl.len()) % 50 == 0))
    ),
    "MLD007": _events(
        lambda f: f.with_columns(
            pl.when(pl.int_range(pl.len()) % 10 < 9).then(pl.lit("s1")).otherwise(pl.col("site")).alias("site")
        )
    ),
    "MLD008": _events(
        lambda f: f.with_columns(
            pl.when(pl.int_range(pl.len()) == 0)
            .then(pl.lit("ghost"))
            .otherwise(pl.col("character_id"))
            .alias("character_id")
        )
    ),
    "MLD009": lambda root: (root / "characters.csv").write_text(
        "character_id,created_at\nc1,2015-01-01\nc2,2014-01-01\nc3,2014-01-01\nc4,2014-01-01\n"
    ),
    "MLD010": _events(lambda f: f.with_columns(pl.lit("s1").alias("site"))),
    "MLR001": _json("manifest.json", lambda m: m.pop("git_sha")),
    "MLR002": lambda root: (root / "events.csv").write_text((root / "events.csv").read_text() + "\n"),
    "MLR003": _json("manifest.json", _set("git_dirty", True)),
    "MLR004": _json("manifest.json", _set("model_seeds", {"dcn": [0]})),
    "MLR005": _json("parity.json", _set("bundle_sha256", "f" * 64)),
    "MLL001": _move_test_row_into_train,
    "MLL002": _overlap_splits_in_time,
    "MLL003": _json("manifest.json", _vocab_fitted_on_test),
    "MLL004": _json("leakage.json", _set("shuffled_label_auc", 0.61)),
    "MLL005": _json("leakage.json", _set("feature_auc", {"clicked_before": 0.97})),
    "MLL007": _json("leakage.json", _set("adversarial_auc", 0.93)),
    "MLM001": _predictions(lambda f: f.filter(~((pl.col("model") == "prior") & (pl.int_range(pl.len()) % 7 == 0)))),
    "MLM002": _predictions(
        lambda f: f.with_columns(pl.when(pl.int_range(pl.len()) == 0).then(1.0).otherwise(pl.col("pred")).alias("pred"))
    ),
    "MLM003": _predictions(
        lambda f: f.with_columns(
            pl.when(_primary_holdout()).then(1 - pl.col("pred")).otherwise(pl.col("pred")).alias("pred")
        )
    ),
    "MLM004": _predictions(
        lambda f: f.with_columns(
            pl.when(pl.col("model") == "dcn").then(pl.lit(0.2)).otherwise(pl.col("pred")).alias("pred")
        )
    ),
    "MLM005": _predictions(_miscalibrate(1.4)),
    "MLM006": _predictions(_miscalibrate(1.4)),
    "MLM007": _predictions(
        lambda f: f.with_columns(
            pl.when(pl.col("model") == "dcn")
            .then((pl.col("pred") - 0.5) * 0.3 + 0.27)
            .otherwise(pl.col("pred"))
            .alias("pred")
        )
    ),
    "MLM008": _predictions(lambda f: f.drop("slice_genre")),
    "MLM009": _predictions(
        lambda f: f.with_columns(
            pl.when(_primary_holdout() & (pl.col("slice_genre") == "horror"))
            .then(1 - pl.col("pred"))
            .otherwise(pl.col("pred"))
            .alias("pred")
        )
    ),
    "MLV001": _json("parity.json", _set("categorical_mismatches", 3)),
    "MLV002": _json("parity.json", _set("onnx_max_abs_diff", 1e-3)),
    "MLV003": _json("latency.json", _set("p99_ms", 71.0)),
    "MLV004": _json("latency.json", _set("error_rate", 0.02)),
    "MLP001": _json("ope.json", lambda o: o["policies"][0].__setitem__("ess", 40)),
    "MLP002": _json("ope.json", lambda o: o["policies"][0].__setitem__("value", 0.5)),
    "MLP003": _serve_gated,
    "MLP004": _single_eligible_not_certain,
    "MLP005": _inflate_served_propensity,
    "MLX001": _json("drift.json", _set("psi", {"site": {"2014-10-26": 0.41}})),
}


def test_every_check_has_a_breakage() -> None:
    assert set(BREAKAGES) == set(CHECKS)


@pytest.mark.parametrize("code", sorted(BREAKAGES))
def test_breakage_is_caught(project: Path, run: Runner, code: str) -> None:
    BREAKAGES[code](project)
    result = run(project, code)
    expected = Status.FAIL if CHECKS[code].severity.value == "error" else Status.WARN
    assert result.status is expected, (result.message, result.details)


def test_missing_artifact_fails_rather_than_passes(project: Path, run: Runner) -> None:
    (project / ART / "latency.json").unlink()
    result = run(project, "MLV003")
    assert result.status is Status.FAIL
    assert "missing artifact" in result.message


def test_missing_source_package_fails(project: Path, run: Runner) -> None:
    (project / "src/demo_pkg/__init__.py").unlink()
    assert run(project, "MLS005").status is Status.FAIL


def test_suppression_comment_silences_finding(project: Path, run: Runner) -> None:
    _append_source("training/train.py", "\nstate = torch.load('model.pt')  # mlcheck: ignore[MLS005]\n")(project)
    assert run(project, "MLS005").status is Status.PASS


def test_unconfigured_section_is_skipped(project: Path, run: Runner) -> None:
    config = (project / "pyproject.toml").read_text()
    (project / "pyproject.toml").write_text(config.split("[tool.mlcheck.data]")[0])
    assert run(project, "MLD001").status is Status.SKIP


def test_notebooks_are_allowed_only_in_configured_directories(project: Path, run: Runner) -> None:
    config = (project / "pyproject.toml").read_text()
    (project / "pyproject.toml").write_text(
        config.replace('package = "demo_pkg"', 'package = "demo_pkg"\nnotebooks_allowed_in = ["experiments"]')
    )
    (project / "experiments" / "E001").mkdir(parents=True)
    (project / "experiments" / "E001" / "notebook.ipynb").write_text("{}")
    assert run(project, "MLS006").status is Status.PASS
    (project / "scratch.ipynb").write_text("{}")
    assert run(project, "MLS006").status is Status.FAIL


def test_a_changed_bundle_fails_evidence_binding(project: Path, run: Runner) -> None:
    """Retraining in place leaves every report naming the old model."""
    (project / ART / "model.bin").write_text("retrained")
    assert run(project, "MLR005").status is Status.FAIL


def test_a_short_load_test_does_not_settle_p99(project: Path, run: Runner) -> None:
    edit_json(project / ART / "latency.json", lambda r: r.update(duration_s=20, requests=4000))
    assert run(project, "MLV003").status is Status.FAIL


def test_shuffled_label_auc_below_chance_is_noise_not_failure(project: Path, run: Runner) -> None:
    edit_json(project / ART / "leakage.json", lambda r: r.update(shuffled_label_auc=0.46))
    assert run(project, "MLL004").status is Status.PASS
