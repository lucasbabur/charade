"""Intrinsic text bake-off (`uv run poe text`): does the description carry CTR signal beyond genre?

For each provider:
    genre_knn_acc    5-NN accuracy predicting genre from the embedding (sanity: > 0.9 or it is broken)
    tier_knn_acc     the same for safety tier
    residual_rho     Spearman correlation, on validation characters, between the observed CTR residual
                     (character CTR minus its genre x tier rate) and a ridge prediction from the embedding
                     trained on training characters. This is the signal text could add beyond metadata.

The extrinsic, decisive test (DCN validation NE with and without text) runs in the model ablations.
"""

import math
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from charade.config import get_settings
from charade.data.load import load_characters, load_joined
from charade.data.split import assign_split
from charade.text.embed import DERIVED_DIR, Provider, derived_path, raw_embeddings, reduce

MIN_IMPRESSIONS = 100
REPORT = Path("reports/text_bakeoff.csv")


def _residuals(frame: pl.DataFrame, split: str, min_impressions: int) -> pl.DataFrame:
    part = frame.filter(pl.col("split") == split)
    cell = part.group_by(["genre", "safety_tier"]).agg(pl.col("click").mean().alias("cell_ctr"))
    return (
        part.group_by("character_id", "genre", "safety_tier")
        .agg(pl.len().alias("n"), pl.col("click").mean().alias("ctr"))
        .filter(pl.col("n") >= min_impressions)
        .join(cell, on=["genre", "safety_tier"])
        .select("character_id", (pl.col("ctr") - pl.col("cell_ctr")).alias("residual"))
    )


def run(
    providers: list[Provider],
    data_dir: Path,
    derived_dir: Path = DERIVED_DIR,
    report: Path = REPORT,
    min_impressions: int = MIN_IMPRESSIONS,
) -> pl.DataFrame:
    """Embed with each provider, write reduced vectors and the bake-off table."""
    settings = get_settings()
    characters = load_characters(data_dir / "characters.csv").sort("character_id")
    frame = assign_split(load_joined(data_dir), settings.train_end, settings.val_end, settings.test_end)
    train_res, val_res = _residuals(frame, "train", min_impressions), _residuals(frame, "val", min_impressions)
    ids = characters["character_id"].to_list()
    index = {c: i for i, c in enumerate(ids)}
    rows: list[dict[str, object]] = []
    derived_dir.mkdir(parents=True, exist_ok=True)
    for provider in providers:
        vectors = raw_embeddings(characters, provider)
        reduce(characters, vectors, settings.train_end).write_parquet(derived_path(provider, derived_dir))
        knn = KNeighborsClassifier(n_neighbors=5)
        genre_acc = cross_val_score(knn, vectors, characters["genre"].to_numpy(), cv=5).mean()
        tier_acc = cross_val_score(knn, vectors, characters["safety_tier"].to_numpy(), cv=5).mean()
        x_train = vectors[[index[c] for c in train_res["character_id"]]]
        x_val = vectors[[index[c] for c in val_res["character_id"]]]
        ridge = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(x_train, train_res["residual"].to_numpy())
        rho = float(spearmanr(ridge.predict(x_val), val_res["residual"].to_numpy()).statistic)  # pyright: ignore[reportAttributeAccessIssue]
        half = 1.96 / math.sqrt(max(val_res.height - 3, 1))  # Fisher z interval
        rows.append(
            {
                "provider": str(provider),
                "dims": vectors.shape[1],
                "genre_knn_acc": float(genre_acc),
                "tier_knn_acc": float(tier_acc),
                "residual_rho": rho,
                "rho_ci_low": math.tanh(math.atanh(rho) - half),
                "rho_ci_high": math.tanh(math.atanh(rho) + half),
                "n_val_characters": val_res.height,
            }
        )
    table = pl.DataFrame(rows)
    report.parent.mkdir(parents=True, exist_ok=True)
    table.write_csv(report)
    return table


if __name__ == "__main__":
    print(run(list(Provider), get_settings().data_dir))
