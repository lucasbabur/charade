"""Shared training steps used by the pipeline, the ablations and the backtest."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch

from charade.config import DcnConfig, Settings
from charade.features.spec import Encoded, FeatureSpec, Group, encode, fit_spec
from charade.models.dataset import build_frame
from charade.models.nets import DCNv2, Ensemble, Logistic
from charade.models.trainer import TrainConfig, TrainResult, predict_logits, train
from charade.text.embed import DIMS, Provider

SPLITS = ("train", "val", "test")


@dataclass
class Prepared:
    """Encoded splits and the spec that produced them."""

    frame: pl.DataFrame
    spec: FeatureSpec
    x: dict[str, Encoded]
    y: dict[str, np.ndarray]


def prepare(frame: pl.DataFrame, groups: set[Group], text: bool) -> Prepared:
    """Fit the spec on train and encode every split."""
    parts = {s: frame.filter(pl.col("split") == s) for s in SPLITS}
    spec = fit_spec(parts["train"], groups | ({Group.TEXT} if text else set()), DIMS if text else 0)
    return Prepared(
        frame=frame,
        spec=spec,
        x={s: encode(spec, p) for s, p in parts.items()},
        y={s: p["click"].to_numpy().astype(np.float64) for s, p in parts.items()},
    )


def load_prepared(
    settings: Settings, data_dir: Path, groups: set[Group] | None = None, text: str | None = None
) -> Prepared:
    """Build the modelling table and encode it with the configured (or given) groups and text provider."""
    chosen = groups if groups is not None else {Group(g) for g in settings.model.groups}
    provider = Provider(text) if text else None
    return prepare(build_frame(settings, data_dir, provider), chosen, provider is not None)


def _id_field(spec: FeatureSpec) -> int | None:
    names = [f.name for f in spec.categorical]
    return names.index("character_id") if "character_id" in names else None


def train_dcn(prep: Prepared, cfg: DcnConfig, seed: int, fixed_steps: int | None = None) -> TrainResult:
    """One DCN-v2 seed, early-stopped on validation."""
    torch.manual_seed(seed)  # weight init is part of the seed
    model = DCNv2(
        prep.spec.cardinalities(),
        prep.x["train"].dense.shape[1],
        cfg.embedding_dim,
        cfg.cross_layers,
        cfg.cross_rank,
        cfg.hidden,
        cfg.dropout,
    )
    config = TrainConfig(
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        batch_size=cfg.batch_size,
        max_epochs=cfg.max_epochs,
        evals_per_epoch=cfg.evals_per_epoch,
        patience=cfg.patience,
        id_dropout=cfg.character_id_dropout,
        id_dropout_field=_id_field(prep.spec),
        seed=seed,
        fixed_steps=fixed_steps,
    )
    return train(model, prep.x["train"], prep.y["train"], prep.x["val"], prep.y["val"], config)


def train_dcn_ensemble(prep: Prepared, cfg: DcnConfig, seeds: list[int]) -> tuple[Ensemble, list[TrainResult]]:
    """Seed ensemble (mean logit); the shipped model."""
    results = [train_dcn(prep, cfg, seed) for seed in seeds]
    return Ensemble([r.model for r in results]), results


def train_logistic(prep: Prepared, seed: int) -> TrainResult:
    """Baseline: logistic regression on the same inputs."""
    torch.manual_seed(seed)
    model = Logistic(prep.spec.cardinalities(), prep.x["train"].dense.shape[1])
    return train(model, prep.x["train"], prep.y["train"], prep.x["val"], prep.y["val"], TrainConfig(lr=3e-3, seed=seed))


def logits(model: object, prep: Prepared, split: str) -> np.ndarray:
    """Logits of a torch model on a split."""
    return predict_logits(model, prep.x[split])  # pyright: ignore[reportArgumentType]
