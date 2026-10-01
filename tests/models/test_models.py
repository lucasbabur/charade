from pathlib import Path

import numpy as np
import onnxruntime as ort
import polars as pl
import pytest
import torch

from charade.config import DcnConfig
from charade.evaluation.metrics import ece, group_auc, normalized_entropy, paired_bootstrap, summary
from charade.features.spec import Group
from charade.models.calibrate import fit_calibrator
from charade.models.core import prepare, train_dcn, train_logistic
from charade.models.export import export_onnx, torch_logits
from charade.models.nets import DCNv2, Ensemble
from charade.scoring.calibration import CalibrationKind, Calibrator

GROUPS = set(Group) - {Group.TEXT}
TINY = DcnConfig(embedding_dim=4, cross_layers=1, cross_rank=8, hidden=[16], max_epochs=2, batch_size=512, seeds=[0])


def test_dcn_learns_on_fixture(features: pl.DataFrame) -> None:
    prep = prepare(features, GROUPS, text=False)
    result = train_dcn(prep, TINY, seed=0)
    assert result.history
    assert result.best_val_logloss < 0.69


def test_logistic_baseline_trains(features: pl.DataFrame) -> None:
    prep = prepare(features, GROUPS, text=False)
    assert train_logistic(prep, seed=0).best_val_logloss < 0.69


def test_training_is_deterministic_on_cpu(features: pl.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("charade.models.trainer.device", lambda: torch.device("cpu"))
    prep = prepare(features, GROUPS, text=False)
    a = train_dcn(prep, TINY, seed=3).best_val_logloss
    b = train_dcn(prep, TINY, seed=3).best_val_logloss
    assert a == b


def test_onnx_export_matches_torch(features: pl.DataFrame, tmp_path: Path) -> None:

    prep = prepare(features, GROUPS, text=False)
    nets: list[torch.nn.Module] = [
        DCNv2(prep.spec.cardinalities(), prep.x["train"].dense.shape[1], 4, 1, 8, [16], 0.0) for _ in range(2)
    ]
    model = Ensemble(nets)
    export_onnx(model, prep.x["test"], tmp_path / "m.onnx")
    session = ort.InferenceSession(str(tmp_path / "m.onnx"), providers=["CPUExecutionProvider"])
    x = prep.x["test"]
    onnx = np.asarray(session.run(["logit"], {"categorical": x.categorical, "dense": x.dense})[0])
    np.testing.assert_allclose(onnx, torch_logits(model, x), atol=1e-5)


def test_calibrator_recovers_platt_shift() -> None:
    rng = np.random.default_rng(0)
    logits = rng.normal(-1.5, 1.0, 50_000)
    y = rng.binomial(1, 1 / (1 + np.exp(-(0.7 * logits - 0.3)))).astype(np.float64)
    calibrator, scores = fit_calibrator(logits, y, np.arange(len(y)) % 24)
    assert calibrator.kind in {CalibrationKind.PLATT, CalibrationKind.ISOTONIC}
    assert scores["identity"] > min(scores.values())
    assert abs(calibrator.apply(logits).mean() - y.mean()) < 0.005


def test_calibrators_roundtrip_and_stay_in_unit_interval() -> None:
    x = np.linspace(-30, 30, 101)
    for c in (
        Calibrator(kind=CalibrationKind.PLATT, a=2, b=1),
        Calibrator(kind=CalibrationKind.ISOTONIC, x=[0, 1], y=[0, 1]),
    ):
        restored = Calibrator.model_validate_json(c.model_dump_json())
        p = restored.apply(x)
        assert ((p > 0) & (p < 1)).all()


def test_metrics_reference_values() -> None:
    y = np.array([0, 0, 1, 1], dtype=np.float64)
    assert normalized_entropy(y, np.full(4, 0.5)) == pytest.approx(1.0)
    assert ece(y, np.array([0.0, 0.0, 1.0, 1.0]), bins=2) == pytest.approx(0.0)
    assert ece(y, np.full(4, 0.5), bins=2) == pytest.approx(0.5)
    assert summary(y, np.array([0.1, 0.2, 0.8, 0.9]))["auc"] == 1.0
    alternating = np.array([0, 1, 0, 1], dtype=np.float64)
    assert group_auc(alternating, np.array([0.1, 0.9, 0.8, 0.2]), np.array([0, 0, 1, 1])) == pytest.approx(0.5)


def test_paired_bootstrap_brackets_mean() -> None:
    rng = np.random.default_rng(1)
    diff = rng.normal(0.01, 0.1, 5000)
    mean, low, high = paired_bootstrap(diff, np.arange(5000) // 100)
    assert low < mean < high
