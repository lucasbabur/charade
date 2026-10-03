"""Leakage probes written to `leakage.json` (mlcheck MLL004, MLL005, MLL007)."""

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from charade.features.counters import RAW_COUNTER_COLUMNS, offline_counters
from charade.features.derive import derive
from charade.features.spec import TEXT_PREFIX
from charade.models.core import Prepared, prepare, train_logistic
from charade.models.trainer import predict_logits

ADVERSARIAL_ROWS = 100_000


def shuffled_label_auc(prep: Prepared, seed: int, repeats: int = 3) -> float:
    """Mean validation AUC of the baseline trained on shuffled clicks; 0.5 if nothing reaches the label.

    Clicks are permuted on the raw rows *before* the label-derived counters are rebuilt, so a leak in
    feature construction (e.g. counters that see the current hour) survives the shuffle and shows up as
    AUC above chance. Validation labels are permuted with the same draw, so early stopping and the score
    both see noise and a clean pipeline lands at 0.5 instead of drifting with the checkpoint chosen.
    """
    raw = prep.frame.drop(RAW_COUNTER_COLUMNS)
    groups, text = (
        set(prep.spec.groups),
        bool(prep.spec.dense) and any(d.startswith(TEXT_PREFIX) for d in prep.spec.dense),
    )
    aucs: list[float] = []
    for repeat in range(repeats):
        rng = np.random.default_rng([seed, repeat])
        shuffled = raw.with_columns(pl.Series("click", rng.permutation(raw["click"].to_numpy())))
        noise = prepare(derive(offline_counters(shuffled)).sort(["ts", "id"]), groups, text)
        result = train_logistic(noise, seed + repeat)
        aucs.append(float(roc_auc_score(noise.y["val"], predict_logits(result.model, noise.x["val"]))))
    return float(np.mean(aucs))


def feature_aucs(prep: Prepared) -> dict[str, float]:
    """Validation AUC of each feature alone: train-split target mean per categorical id, raw value for dense."""
    y_train, y_val = prep.y["train"], prep.y["val"]
    out: dict[str, float] = {}
    for i, field in enumerate(prep.spec.categorical):
        ids_train, ids_val = prep.x["train"].categorical[:, i], prep.x["val"].categorical[:, i]
        size = int(max(ids_train.max(), ids_val.max())) + 1
        clicks = np.bincount(ids_train, weights=y_train, minlength=size)
        counts = np.bincount(ids_train, minlength=size)
        rate = (clicks + y_train.mean() * 20) / (counts + 20)
        out[field.name] = float(roc_auc_score(y_val, rate[ids_val]))
    for j, name in enumerate(prep.spec.dense):
        out[name] = float(roc_auc_score(y_val, prep.x["val"].dense[:, j]))
    return out


def adversarial_auc(prep: Prepared, seed: int) -> float:
    """Hold-out AUC of a classifier separating training rows from test rows (covariate shift)."""
    rng = np.random.default_rng(seed)

    def sample(split: str) -> np.ndarray:
        x = np.hstack([prep.x[split].categorical, prep.x[split].dense])
        return x[rng.choice(len(x), size=min(ADVERSARIAL_ROWS, len(x)), replace=False)]

    a, b = sample("train"), sample("test")
    x = np.vstack([a, b])
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    order = rng.permutation(len(y))
    x, y = x[order], y[order]
    half = len(y) // 2
    categorical = list(range(len(prep.spec.categorical)))
    params = {
        "objective": "binary",
        "num_leaves": 63,
        "learning_rate": 0.1,
        "verbose": -1,
        "seed": seed,
        "num_threads": 16,
    }
    model = lgb.train(params, lgb.Dataset(x[:half], y[:half], categorical_feature=categorical), 200)
    return float(roc_auc_score(y[half:], model.predict(x[half:])))
