import json
from pathlib import Path

import pytest

from charade.config import DcnConfig, GbdtConfig, ModelConfig, Settings, get_settings
from charade.models import ablate, pipeline
from charade.scoring.scorer import Scorer
from tests.conftest import FIXTURES

TINY = ModelConfig(
    dcn=DcnConfig(
        embedding_dim=4, cross_layers=1, cross_rank=8, hidden=[16], max_epochs=2, batch_size=512, seeds=[0, 1, 2]
    ),
    gbdt=GbdtConfig(num_leaves=7, max_rounds=30, min_data_in_leaf=20),
)


def _settings(tmp_path: Path) -> Settings:
    return get_settings().model_copy(update={"data_dir": FIXTURES, "artifacts_dir": tmp_path / "art", "model": TINY})


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
        "evaluation_ledger.jsonl",
    ):
        assert (art / name).is_file(), name
    manifest = json.loads((art / "manifest.json").read_text())
    assert {a["fit_split"] for a in manifest["fitted_artifacts"]} == {"train", "val"}
    assert metrics["onnx_max_abs_diff"] < 1e-4  # type: ignore[operator]
    assert Scorer(art).spec.categorical


def test_ledger_is_appended_not_truncated(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    pipeline.run(settings, reports=tmp_path / "reports")
    pipeline.run(settings, reports=tmp_path / "reports")
    lines = (settings.artifacts_dir / "evaluation_ledger.jsonl").read_text().splitlines()
    assert len(lines) == 6
    assert len({json.loads(line)["run_id"] for line in lines}) == 2


def test_ablations_compare_against_full_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ablate, "get_settings", lambda: _settings(tmp_path))
    variants = (ablate.VARIANTS[0], ablate.VARIANTS[4], ablate.VARIANTS[8])
    table = ablate.run(variants, tmp_path, FIXTURES)
    assert table["variant"].to_list() == ["all features", "- user_history", "+ text (tfidf)"]
    assert table["delta_logloss"][0] == 0.0
