"""LightGBM yardstick on the same encoded features (categorical ids as native categoricals)."""

import lightgbm as lgb
import numpy as np

from charade.config import GbdtConfig
from charade.features.spec import Encoded, FeatureSpec


def _matrix(data: Encoded) -> np.ndarray:
    return np.hstack([data.categorical.astype(np.float64), data.dense.astype(np.float64)])


def train_gbdt(
    spec: FeatureSpec,
    train: Encoded,
    y_train: np.ndarray,
    val: Encoded,
    y_val: np.ndarray,
    config: GbdtConfig,
    seed: int = 0,
) -> lgb.Booster:
    """Fit with early stopping on validation log loss."""
    names = [f.name for f in spec.categorical] + spec.dense
    categorical = list(range(len(spec.categorical)))
    train_set = lgb.Dataset(_matrix(train), y_train, feature_name=names, categorical_feature=categorical)
    val_set = lgb.Dataset(_matrix(val), y_val, reference=train_set)
    params = {
        "objective": "binary",
        "learning_rate": config.learning_rate,
        "num_leaves": config.num_leaves,
        "min_data_in_leaf": config.min_data_in_leaf,
        "cat_smooth": config.cat_smooth,
        "cat_l2": config.cat_l2,
        "max_cat_to_onehot": 8,
        "feature_fraction": config.feature_fraction,
        "lambda_l2": config.lambda_l2,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "seed": seed,
        "deterministic": True,
        "num_threads": 16,
        "verbose": -1,
    }
    return lgb.train(
        params, train_set, config.max_rounds, valid_sets=[val_set], callbacks=[lgb.early_stopping(100, verbose=False)]
    )


def predict_gbdt(model: lgb.Booster, data: Encoded) -> np.ndarray:
    """Probabilities at the best iteration."""
    return np.asarray(model.predict(_matrix(data), num_iteration=model.best_iteration), dtype=np.float64)
