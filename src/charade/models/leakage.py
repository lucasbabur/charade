"""Leakage probes written to `leakage.json` (mlcheck MLL004, MLL005, MLL007)."""

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score

from charade.models.core import Prepared, train_logistic
from charade.models.trainer import predict_logits

ADVERSARIAL_ROWS = 100_000


def shuffled_label_auc(prep: Prepared, seed: int) -> float:
    """Validation AUC of the baseline retrained on permuted training labels (chance if nothing leaks)."""
    rng = np.random.default_rng(seed)
    shuffled = Prepared(prep.frame, prep.spec, prep.x, {**prep.y, "train": rng.permutation(prep.y["train"])})
    result = train_logistic(shuffled, seed)
    return float(roc_auc_score(prep.y["val"], predict_logits(result.model, prep.x["val"])))


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
