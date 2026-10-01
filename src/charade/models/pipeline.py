"""End-to-end training run (`uv run poe train`): the artifact contract checked by mlcheck.

Writes to `[tool.charade].artifacts_dir`:
    model.onnx, feature_spec.json, calibrator.json, characters.parquet    the serving bundle
    manifest.json, splits.parquet, predictions.parquet, leakage.json      evidence for mlcheck
    evaluation_ledger.jsonl                                               appended, never truncated
and reports/models/metrics.md (+ metrics.json) with every number the docs cite.

The test split is scored once per run, for every model, after all choices were made on validation.
"""

import hashlib
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import polars as pl
import torch

from charade.config import Settings, get_settings
from charade.data.load import load_characters
from charade.evaluation.metrics import logloss_rows, paired_bootstrap, summary
from charade.evaluation.slices import SLICES, add_slices
from charade.features.spec import Encoded
from charade.models import leakage
from charade.models.calibrate import fit_calibrator
from charade.models.core import SPLITS, Prepared, load_prepared, logits, train_dcn_ensemble, train_logistic
from charade.models.export import export_onnx, torch_logits
from charade.models.gbdt import predict_gbdt, train_gbdt
from charade.ranking.evidence import build_evidence
from charade.scoring.scorer import CALIBRATOR_FILE, EVIDENCE_FILE, MODEL_FILE, SPEC_FILE, Scorer
from charade.text.embed import Provider, derived_path

REPORTS = Path("reports/models")
PRIMARY, BASELINE, GBDT = "dcn_v2", "logreg", "lightgbm"


def _git() -> tuple[str, bool]:
    """(commit, dirty). Images carry the commit in CHARADE_GIT_SHA (no .git inside); builds are clean."""
    baked = os.environ.get("CHARADE_GIT_SHA")
    if baked:
        return baked, False

    # Fixed argv, no shell, no user input: git is resolved from PATH on purpose.
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()  # noqa: S603, S607

    return git("rev-parse", "HEAD"), bool(git("status", "--porcelain", "--untracked-files=no"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def _windows(frame: pl.DataFrame) -> dict[str, dict[str, str]]:
    bounds = frame.group_by("split").agg(pl.col("ts").min().alias("start"), pl.col("ts").max().alias("end"))
    return {
        r["split"]: {"start": r["start"].isoformat(), "end": r["end"].isoformat()} for r in bounds.iter_rows(named=True)
    }


def _hours(frame: pl.DataFrame, split: str) -> np.ndarray:
    return frame.filter(pl.col("split") == split)["ts"].dt.hour().to_numpy()


def _prediction_frame(prep: Prepared, model: str, preds: dict[str, np.ndarray]) -> pl.DataFrame:
    sliced = add_slices(prep.frame)
    parts = []
    for split in ("val", "test"):
        rows = sliced.filter(pl.col("split") == split)
        parts.append(
            rows.select(
                "id",
                "split",
                pl.lit(model).alias("model"),
                pl.col("click").cast(pl.Int64).alias("label"),
                "ts",
                *[f"slice_{s}" for s in SLICES],
            ).with_columns(pl.Series("pred", preds[split]))
        )
    return pl.concat(parts)


def _comparisons(prep: Prepared, preds: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, float]]:
    """Paired hour-block bootstrap of test log loss: primary minus each other model."""
    y = prep.y["test"]
    blocks = prep.frame.filter(pl.col("split") == "test")["ts"].dt.epoch("s").to_numpy() // 3600
    out: dict[str, dict[str, float]] = {}
    for other in (BASELINE, GBDT):
        diff = logloss_rows(y, preds[PRIMARY]["test"]) - logloss_rows(y, preds[other]["test"])
        mean, low, high = paired_bootstrap(diff, blocks)
        out[f"{PRIMARY}-{other}"] = {"delta_logloss": mean, "ci_low": low, "ci_high": high}
    return out


def _slice_table(prep: Prepared, p: np.ndarray) -> list[dict[str, object]]:
    rows = add_slices(prep.frame).filter(pl.col("split") == "test").with_columns(pl.Series("p", p))
    table: list[dict[str, object]] = []
    for name in SLICES:
        for (value,), part in rows.group_by(f"slice_{name}"):
            y, q = part["click"].to_numpy().astype(np.float64), part["p"].to_numpy()
            if len(y) >= 500 and 0 < y.sum() < len(y):
                table.append({"slice": name, "value": str(value), **summary(y, q)})
    return sorted(table, key=lambda r: (str(r["slice"]), str(r["value"])))


def _character_table(settings: Settings, provider: str | None) -> pl.DataFrame:
    characters = load_characters(settings.data_dir / "characters.csv")
    if provider:
        characters = characters.join(pl.read_parquet(derived_path(Provider(provider))), on="character_id", how="left")
    return characters


def run(settings: Settings | None = None, out: Path | None = None, reports: Path = REPORTS) -> dict[str, object]:
    """Train every model, calibrate, evaluate once on test, export, and write the evidence."""
    settings = settings or get_settings()
    out = out or settings.artifacts_dir
    out.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    git_sha, dirty = _git()
    cfg = settings.model
    prep = load_prepared(settings, settings.data_dir, text=cfg.text_provider)

    ensemble, seed_results = train_dcn_ensemble(prep, cfg.dcn, cfg.dcn.seeds)
    baseline = train_logistic(prep, settings.seed)
    gbdt = train_gbdt(
        prep.spec, prep.x["train"], prep.y["train"], prep.x["val"], prep.y["val"], cfg.gbdt, settings.seed
    )

    raw = {
        PRIMARY: {s: logits(ensemble, prep, s) for s in ("val", "test")},
        BASELINE: {s: logits(baseline.model, prep, s) for s in ("val", "test")},
        GBDT: {s: _logit(predict_gbdt(gbdt, prep.x[s])) for s in ("val", "test")},
    }
    calibrators, calibration_scores, preds = {}, {}, {}
    for name, by_split in raw.items():
        calibrator, scores = fit_calibrator(
            by_split["val"], prep.y["val"], _hours(prep.frame, "val")
        )  # mlcheck: ignore[MLS007]
        calibrators[name], calibration_scores[name] = calibrator, scores
        preds[name] = {s: calibrator.apply(v) for s, v in by_split.items()}
    seed_val = {
        str(seed): summary(prep.y["val"], 1 / (1 + np.exp(-logits(r.model, prep, "val"))))["ne"]
        for seed, r in zip(cfg.dcn.seeds, seed_results, strict=True)
    }

    # Serving bundle.
    export_onnx(ensemble, prep.x["test"], out / MODEL_FILE)
    (out / SPEC_FILE).write_text(prep.spec.model_dump_json())
    (out / CALIBRATOR_FILE).write_text(calibrators[PRIMARY].model_dump_json())
    _character_table(settings, cfg.text_provider).write_parquet(out / "characters.parquet")
    (out / EVIDENCE_FILE).write_text(build_evidence(prep.frame.filter(pl.col("split") == "train")).model_dump_json())
    scorer = Scorer(out)
    sample = Encoded(categorical=prep.x["test"].categorical[:20_000], dense=prep.x["test"].dense[:20_000])
    onnx_diff = float(np.abs(scorer.logits(sample) - torch_logits(ensemble, sample)).max())

    # Evidence for mlcheck.
    run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{git_sha[:8]}"
    prep.frame.select("id", "split", "ts").write_parquet(out / "splits.parquet")
    pl.concat([_prediction_frame(prep, m, p) for m, p in preds.items()]).write_parquet(out / "predictions.parquet")
    leakage_report = {
        "shuffled_label_auc": leakage.shuffled_label_auc(prep, settings.seed),
        "feature_auc": leakage.feature_aucs(prep),
        "adversarial_auc": leakage.adversarial_auc(prep, settings.seed),
    }
    (out / "leakage.json").write_text(json.dumps(leakage_report, indent=2))
    windows = _windows(prep.frame)
    train_end, val_end = windows["train"]["end"], windows["val"]["end"]
    fitted = [
        {"name": "feature_spec(vocabularies, dense scaling)", "fit_split": "train", "fit_end": train_end},
        *({"name": f"dcn_v2 seed {s}", "fit_split": "train", "fit_end": train_end} for s in cfg.dcn.seeds),
        {"name": "logreg", "fit_split": "train", "fit_end": train_end},
        {"name": "lightgbm", "fit_split": "train", "fit_end": train_end},
        *({"name": f"calibrator {m}", "fit_split": "val", "fit_end": val_end} for m in preds),
    ]
    if cfg.text_provider:
        fitted.append({"name": f"text PCA ({cfg.text_provider})", "fit_split": "train", "fit_end": train_end})
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "git_sha": git_sha,
        "git_dirty": dirty,
        "config_hash": hashlib.sha256(settings.model.model_dump_json().encode()).hexdigest()[:16],
        "data_sha256": {
            str(settings.data_dir / f): _sha256(settings.data_dir / f) for f in ("impressions.csv", "characters.csv")
        },
        "windows": windows,
        "fitted_artifacts": fitted,
        "model_seeds": {PRIMARY: cfg.dcn.seeds, BASELINE: [settings.seed], GBDT: [settings.seed]},
        "library_versions": {p: version(p) for p in ("torch", "lightgbm", "polars", "onnxruntime", "scikit-learn")},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    with (out / "evaluation_ledger.jsonl").open("a") as ledger:
        for model in preds:
            entry = {
                "timestamp": datetime.now(UTC).isoformat(),
                "run_id": run_id,
                "model_version": f"{model}@{run_id}",
                "split": "test",
            }
            ledger.write(json.dumps(entry) + "\n")

    metrics: dict[str, object] = {
        "run_id": run_id,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "seconds": time.monotonic() - started,
        "config": cfg.model_dump(),
        "models": {m: {s: summary(prep.y[s], p[s]) for s in ("val", "test")} for m, p in preds.items()},
        "uncalibrated_test": {m: summary(prep.y["test"], 1 / (1 + np.exp(-v["test"]))) for m, v in raw.items()},
        "calibration_choice": {
            m: {"kind": str(c.kind), "cv_logloss": calibration_scores[m]} for m, c in calibrators.items()
        },
        "dcn_seed_val_ne_uncalibrated": seed_val,
        "dcn_best_steps": [r.best_step for r in seed_results],
        "comparisons_test": _comparisons(prep, preds),
        "slices_test": _slice_table(prep, preds[PRIMARY]["test"]),
        "onnx_max_abs_diff": onnx_diff,
        "rows": {s: len(prep.y[s]) for s in SPLITS},
    }
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    (out / "export.json").write_text(json.dumps({"onnx_max_abs_diff": onnx_diff}))
    return metrics


if __name__ == "__main__":
    run()
