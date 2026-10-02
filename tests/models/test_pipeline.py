import json
from pathlib import Path

import pytest

from charade.analysis import importance
from charade.config import Settings
from charade.models import ablate, pipeline
from charade.scoring.scorer import Scorer
from tests.conftest import FIXTURES, fixture_settings


def _settings(tmp_path: Path) -> Settings:
    return fixture_settings(tmp_path / "art")


def test_pipeline_writes_the_mlcheck_contract(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    metrics = pipeline.run(settings, reports=tmp_path / "reports")
    art = settings.artifacts_dir
    for name in (
        "model.onnx",
        "feature_spec.json",
        "calibrator.json",
        "characters.parquet",
        "manifest.json",
        "splits.parquet",
        "predictions.parquet",
        "leakage.json",
    ):
        assert (art / name).is_file(), name
    manifest = json.loads((art / "manifest.json").read_text())
    assert {a["fit_split"] for a in manifest["fitted_artifacts"]} == {"train", "val"}
    assert metrics["onnx_max_abs_diff"] < 1e-4  # type: ignore[operator]
    assert Scorer(art).spec.categorical


def test_ablations_compare_against_full_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ablate, "get_settings", lambda: _settings(tmp_path))
    variants = (ablate.VARIANTS[0], ablate.VARIANTS[4], ablate.VARIANTS[8])
    table = ablate.run(variants, tmp_path, FIXTURES)
    assert table["variant"].to_list() == ["all features", "- user_history", "+ text (tfidf)"]
    assert table["delta_logloss"][0] == 0.0


def test_importance_covers_every_feature_with_intervals(tmp_path: Path) -> None:
    table = importance.run(_settings(tmp_path), FIXTURES, tmp_path)
    assert table.height == len(set(table["feature"]))
    assert {"character_id", "genre", "log_turn"} <= set(table["feature"])
    assert (table["ci_low"] <= table["permutation_delta"]).all()
    assert (table["mean_abs_shap"] >= 0).all()
    assert (tmp_path / "importance.csv").is_file()


def test_baked_commit_is_used_inside_images(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHARADE_GIT_SHA", "a" * 40)
    assert pipeline._git() == ("a" * 40, False)  # pyright: ignore[reportPrivateUsage]


def test_generated_reports_do_not_make_the_tree_dirty(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import subprocess  # noqa: PLC0415

    monkeypatch.delenv("CHARADE_GIT_SHA", raising=False)
    monkeypatch.chdir(tmp_path)

    def git(*args: str) -> None:
        subprocess.run(["git", *args], check=True, capture_output=True)  # noqa: S603, S607

    git("init", "-q")
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "init")
    (tmp_path / "reports").mkdir()
    (tmp_path / "src.py").write_text("a = 1\n")
    (tmp_path / "reports" / "metrics.json").write_text("{}\n")
    git("add", ".")
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "files")
    (tmp_path / "reports" / "metrics.json").write_text('{"new": 1}\n')
    assert pipeline._git()[1] is False  # pyright: ignore[reportPrivateUsage]
    (tmp_path / "src.py").write_text("a = 2\n")
    assert pipeline._git()[1] is True  # pyright: ignore[reportPrivateUsage]
