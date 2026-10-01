"""Refit study (`uv run poe refit-study`): should the shipped model also train on the validation day?

Early stopping and calibration need a held-out day, so the shipped model never sees 10-28, while the
staleness backtest says each missing day costs ~0.003 NE. A refit (train on train + val for the
step count early stopping chose, reuse the Platt map fitted on val) could recover that. The
decision is made one day earlier, without touching test:

    early-stopped:  train 10-21..26, early-stop + calibrate on 10-27, score 10-28
    refit:          then retrain on 10-21..27 for the same steps per seed, same calibrator, score 10-28

Both are 3-seed ensembles; the paired hour-block bootstrap on 10-28 decides.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from charade.config import get_settings
from charade.evaluation.metrics import logloss_rows, normalized_entropy, paired_bootstrap
from charade.features.spec import Group, encode, fit_spec
from charade.models.calibrate import fit_calibrator
from charade.models.core import Prepared, train_dcn
from charade.models.dataset import build_frame
from charade.models.nets import Ensemble
from charade.models.trainer import predict_logits

INNER_TRAIN_END = datetime(2014, 10, 26, 23)
INNER_VAL_END = datetime(2014, 10, 27, 23)
OUT = Path("reports/models/refit_study.json")


def _prepared(frame: pl.DataFrame, train_end: datetime, groups: set[Group]) -> Prepared:
    split = frame.with_columns(
        pl.when(pl.col("ts") <= train_end).then(pl.lit("train")).otherwise(pl.lit("val")).alias("split")
    )
    train, val = split.filter(pl.col("split") == "train"), split.filter(pl.col("split") == "val")
    spec = fit_spec(train, groups)
    x = {"train": encode(spec, train), "val": encode(spec, val)}
    y = {"train": train["click"].to_numpy().astype(np.float64), "val": val["click"].to_numpy().astype(np.float64)}
    return Prepared(frame=split, spec=spec, x=x, y=y)


def run(data_dir: Path | None = None, out: Path = OUT) -> dict[str, object]:
    """Compare early-stopped vs refit ensembles on the validation day; write the result."""
    settings = get_settings()
    cfg, groups = settings.model.dcn, {Group(g) for g in settings.model.groups}
    frame = build_frame(settings, data_dir or settings.data_dir)
    upto_val = frame.filter(pl.col("ts") <= INNER_VAL_END)
    target = frame.filter((pl.col("ts") > INNER_VAL_END) & (pl.col("ts") <= settings.val_end))
    early = _prepared(upto_val, INNER_TRAIN_END, groups)
    refit = _prepared(upto_val, INNER_VAL_END, groups)
    results = [train_dcn(early, cfg, seed) for seed in cfg.seeds]
    stopped = Ensemble([r.model for r in results])
    calibrator, _ = fit_calibrator(
        predict_logits(stopped, early.x["val"]),
        early.y["val"],
        early.frame.filter(pl.col("split") == "val")["ts"].dt.hour().to_numpy(),
    )
    refit_cfg = cfg.model_copy()
    refit_models = [
        train_dcn(refit, refit_cfg, seed, fixed_steps=r.best_step).model
        for seed, r in zip(cfg.seeds, results, strict=True)
    ]
    y = target["click"].to_numpy().astype(np.float64)
    p_early = calibrator.apply(predict_logits(stopped, encode(early.spec, target)))
    p_refit = calibrator.apply(predict_logits(Ensemble(refit_models), encode(refit.spec, target)))
    blocks = target["ts"].dt.hour().to_numpy()
    delta, low, high = paired_bootstrap(logloss_rows(y, p_refit) - logloss_rows(y, p_early), blocks)
    report: dict[str, object] = {
        "scored_on": "2014-10-28 (validation day; test untouched)",
        "early_stopped_ne": normalized_entropy(y, p_early),
        "refit_ne": normalized_entropy(y, p_refit),
        "refit_minus_early_logloss": delta,
        "ci_low": low,
        "ci_high": high,
        "early_calibration_ratio": float(p_early.mean() / y.mean()),
        "refit_calibration_ratio": float(p_refit.mean() / y.mean()),
        "steps": float(np.mean([r.best_step for r in results])),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    print(run())
