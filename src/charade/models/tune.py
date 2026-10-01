"""Hyperparameter search (`uv run poe tune`).

Searches on an **inner split**: train 10-21..10-26, select on 10-27. The real validation day (10-28)
then stays clean for early stopping, calibration and model comparison, and test is untouched.
The winners are copied into `[tool.charade.model]` by hand, citing reports/tuning/*.csv.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import polars as pl
import torch

from charade.config import DcnConfig, GbdtConfig, get_settings
from charade.evaluation.metrics import normalized_entropy
from charade.features.spec import Group, encode, fit_spec
from charade.models.dataset import build_frame
from charade.models.gbdt import predict_gbdt, train_gbdt
from charade.models.nets import DCNv2
from charade.models.trainer import TrainConfig, predict_logits, train

INNER_TRAIN_END = datetime(2014, 10, 26, 23)
OUT = Path("reports/tuning")


def _inner(frame: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    train = frame.filter(pl.col("split") == "train")
    return train.filter(pl.col("ts") <= INNER_TRAIN_END), train.filter(pl.col("ts") > INNER_TRAIN_END)


def dcn_config(trial: optuna.Trial) -> DcnConfig:
    """Search space for DCN-v2."""
    width = trial.suggest_categorical("hidden", ["256-128", "512-256", "400-400-400", "128-64"])
    return DcnConfig(
        embedding_dim=trial.suggest_categorical("embedding_dim", [8, 16, 24, 32]),
        cross_layers=trial.suggest_int("cross_layers", 1, 4),
        cross_rank=trial.suggest_categorical("cross_rank", [32, 64, 128]),
        hidden=[int(w) for w in width.split("-")],
        dropout=trial.suggest_float("dropout", 0.0, 0.4),
        lr=trial.suggest_float("lr", 2e-4, 3e-3, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-7, 1e-1, log=True),
        batch_size=trial.suggest_categorical("batch_size", [1024, 2048, 4096, 8192]),
        character_id_dropout=trial.suggest_float("character_id_dropout", 0.0, 0.3),
    )


def gbdt_config(trial: optuna.Trial) -> GbdtConfig:
    """Search space for LightGBM."""
    return GbdtConfig(
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        num_leaves=trial.suggest_int("num_leaves", 31, 511, log=True),
        min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 50, 2000, log=True),
        cat_smooth=trial.suggest_float("cat_smooth", 1, 200, log=True),
        cat_l2=trial.suggest_float("cat_l2", 1, 100, log=True),
        feature_fraction=trial.suggest_float("feature_fraction", 0.4, 1.0),
        lambda_l2=trial.suggest_float("lambda_l2", 1e-3, 100, log=True),
    )


def run(trials_dcn: int = 60, trials_gbdt: int = 40, data_dir: Path | None = None, out: Path = OUT) -> None:
    """Run both searches and write one CSV per model."""
    settings = get_settings()
    groups = {Group(g) for g in settings.model.groups}
    frame = build_frame(settings, data_dir or settings.data_dir)
    inner_train, inner_val = _inner(frame)
    spec = fit_spec(inner_train, groups)
    x_train, x_val = encode(spec, inner_train), encode(spec, inner_val)
    y_train, y_val = inner_train["click"].to_numpy(), inner_val["click"].to_numpy()
    id_field = spec.field_index("character_id") if Group.CHARACTER_ID in groups else None

    def dcn_objective(trial: optuna.Trial) -> float:
        cfg = dcn_config(trial)
        torch.manual_seed(settings.seed)  # before construction: weight init is part of the trial
        model = DCNv2(
            spec.cardinalities(),
            x_train.dense.shape[1],
            cfg.embedding_dim,
            cfg.cross_layers,
            cfg.cross_rank,
            cfg.hidden,
            cfg.dropout,
        )
        tc = TrainConfig(
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            batch_size=cfg.batch_size,
            max_epochs=cfg.max_epochs,
            id_dropout=cfg.character_id_dropout,
            id_dropout_field=id_field,
            seed=settings.seed,
        )
        result = train(model, x_train, y_train, x_val, y_val, tc)
        trial.set_user_attr("best_step", result.best_step)
        return normalized_entropy(y_val.astype(np.float64), 1 / (1 + np.exp(-predict_logits(result.model, x_val))))

    def gbdt_objective(trial: optuna.Trial) -> float:
        model = train_gbdt(spec, x_train, y_train, x_val, y_val, gbdt_config(trial), seed=settings.seed)
        trial.set_user_attr("best_iteration", model.best_iteration)
        return normalized_entropy(y_val.astype(np.float64), predict_gbdt(model, x_val))

    out.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for name, objective, n in (("dcn", dcn_objective, trials_dcn), ("gbdt", gbdt_objective, trials_gbdt)):
        if n == 0:
            continue  # keep the committed search for this model
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=settings.seed))
        study.optimize(objective, n_trials=n)
        rows = [
            {"trial": t.number, "value": t.value, **t.params, **t.user_attrs}
            for t in study.trials
            if t.value is not None
        ]
        pl.DataFrame(rows).sort("value").write_csv(out / f"{name}.csv")


if __name__ == "__main__":
    import sys

    # `python -m charade.models.tune [dcn_trials gbdt_trials]`; 60/60 gives both models equal search effort.
    counts = [int(a) for a in sys.argv[1:3]] or [60, 60]
    run(trials_dcn=counts[0], trials_gbdt=counts[1])
