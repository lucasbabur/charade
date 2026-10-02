"""Per-feature importance on a LightGBM fit (`python -m charade.analysis.importance`) -> reports/models/importance.csv.

Ablations decide which feature *groups* ship; this explains individual features. One LightGBM (the
tuned config) is fitted on train with every non-text group, including the two the ablations dropped
(character ID, conversation), and read on the validation day two ways:
    mean |SHAP|      LightGBM's exact TreeSHAP contributions (`pred_contrib`), in logit units: how much
                     the model *uses* a feature
    permutation Δ    validation log-loss increase when the feature is shuffled across rows, with a
                     95 % hour-block bootstrap CI: how much the model *needs* it
A feature can be used heavily yet be replaceable (correlated with another), so neither number selects
features on its own; selection stays with the group ablations.
"""

from pathlib import Path

import numpy as np
import polars as pl

from charade.config import Settings, get_settings
from charade.evaluation.metrics import logloss_rows, paired_bootstrap
from charade.features.spec import DENSE_FIELDS, Encoded, Group
from charade.models.core import load_prepared
from charade.models.gbdt import predict_gbdt, train_gbdt

OUT = Path("reports/models")
SHAP_ROWS = 50_000


def run(settings: Settings | None = None, data_dir: Path | None = None, out: Path = OUT) -> pl.DataFrame:
    """Fit, explain on validation, write `importance.csv`, return the table sorted by permutation Δ."""
    settings = settings or get_settings()
    groups = set(Group) - {Group.TEXT}
    prep = load_prepared(settings, data_dir or settings.data_dir, groups=groups)
    model = train_gbdt(
        prep.spec, prep.x["train"], prep.y["train"], prep.x["val"], prep.y["val"], settings.model.gbdt, settings.seed
    )
    val, y = prep.x["val"], prep.y["val"]
    hours = prep.frame.filter(pl.col("split") == "val")["ts"].dt.epoch("s").to_numpy() // 3600
    rng = np.random.default_rng(settings.seed)
    sample = rng.choice(len(y), size=min(SHAP_ROWS, len(y)), replace=False)
    matrix = np.hstack([val.categorical[sample].astype(np.float64), val.dense[sample].astype(np.float64)])
    shap = np.abs(np.asarray(model.predict(matrix, pred_contrib=True))[:, :-1]).mean(axis=0)  # last column: bias
    base = logloss_rows(y, predict_gbdt(model, val))
    names = [f.name for f in prep.spec.categorical] + prep.spec.dense
    groups_of = [str(f.group) for f in prep.spec.categorical] + [str(DENSE_FIELDS[d]) for d in prep.spec.dense]
    n_cat = len(prep.spec.categorical)
    rows: list[dict[str, object]] = []
    for i, (name, group) in enumerate(zip(names, groups_of, strict=True)):
        categorical, dense = val.categorical.copy(), val.dense.copy()
        if i < n_cat:
            categorical[:, i] = rng.permutation(categorical[:, i])
        else:
            dense[:, i - n_cat] = rng.permutation(dense[:, i - n_cat])
        shuffled = logloss_rows(y, predict_gbdt(model, Encoded(categorical=categorical, dense=dense)))
        delta, low, high = paired_bootstrap(shuffled - base, hours)
        rows.append(
            {
                "feature": name,
                "group": group,
                "mean_abs_shap": float(shap[i]),
                "permutation_delta": delta,
                "ci_low": low,
                "ci_high": high,
            }
        )
    table = pl.DataFrame(rows).sort("permutation_delta", descending=True)
    out.mkdir(parents=True, exist_ok=True)
    table.write_csv(out / "importance.csv")
    return table


if __name__ == "__main__":
    print(run())
