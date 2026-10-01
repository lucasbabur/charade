"""Feature spec: which columns feed the model, their vocabularies and dense scaling.

Fitted on the training split only, serialized to JSON, and loaded by serving. Index 0 of every
vocabulary is the out-of-vocabulary bucket: unseen and rare values (count < min_count) share it.
"""

from enum import StrEnum

import numpy as np
import numpy.typing as npt
import polars as pl
from pydantic import BaseModel, Field

from charade.features.counters import COUNTER_FEATURES


class Group(StrEnum):
    """Feature groups; ablations drop whole groups."""

    CONTEXT = "context"
    DEVICE = "device"
    AD = "ad"
    CHARACTER_META = "character_meta"
    CHARACTER_ID = "character_id"
    TEXT = "text"
    CONVERSATION = "conversation"
    USER_HISTORY = "user_history"


class CategoricalField(BaseModel):
    """One embedding field."""

    name: str
    group: Group
    min_count: int = 5
    candidate: bool = False
    """True if the value comes from the candidate ad rather than the opportunity context."""


CATEGORICAL_FIELDS: tuple[CategoricalField, ...] = (
    CategoricalField(name="hour_of_day", group=Group.CONTEXT),
    CategoricalField(name="site_id", group=Group.CONTEXT),
    CategoricalField(name="site_domain", group=Group.CONTEXT),
    CategoricalField(name="site_category", group=Group.CONTEXT),
    CategoricalField(name="app_id", group=Group.CONTEXT),
    CategoricalField(name="app_domain", group=Group.CONTEXT),
    CategoricalField(name="app_category", group=Group.CONTEXT),
    CategoricalField(name="C1", group=Group.CONTEXT),
    CategoricalField(name="C20", group=Group.CONTEXT),
    CategoricalField(name="device_model", group=Group.DEVICE),
    CategoricalField(name="device_type", group=Group.DEVICE),
    CategoricalField(name="device_conn_type", group=Group.DEVICE),
    CategoricalField(name="device_id_real", group=Group.DEVICE, min_count=20),
    CategoricalField(name="banner_pos", group=Group.AD, candidate=True),
    CategoricalField(name="C14", group=Group.AD, candidate=True),
    CategoricalField(name="C15", group=Group.AD, candidate=True),
    CategoricalField(name="C16", group=Group.AD, candidate=True),
    CategoricalField(name="C17", group=Group.AD, candidate=True),
    CategoricalField(name="C18", group=Group.AD, candidate=True),
    CategoricalField(name="C19", group=Group.AD, candidate=True),
    CategoricalField(name="C21", group=Group.AD, candidate=True),
    CategoricalField(name="genre", group=Group.CHARACTER_META),
    CategoricalField(name="safety_tier", group=Group.CHARACTER_META),
    CategoricalField(name="creator_type", group=Group.CHARACTER_META),
    CategoricalField(name="interactions_bucket", group=Group.CHARACTER_META),
    CategoricalField(name="character_id", group=Group.CHARACTER_ID, min_count=20),
    CategoricalField(name="turn_bucket", group=Group.CONVERSATION),
    CategoricalField(name="session_bucket", group=Group.CONVERSATION),
)

DENSE_FIELDS: dict[str, Group] = {
    **dict.fromkeys(COUNTER_FEATURES, Group.USER_HISTORY),
    "log_num_interactions": Group.CHARACTER_META,
    "log_character_age_days": Group.CHARACTER_META,
    "log_turn": Group.CONVERSATION,
    "turn_position": Group.CONVERSATION,
}

TEXT_PREFIX = "text_"
SMALL_FRAME = 2000


class FeatureSpec(BaseModel):
    """Fitted feature spec (serialized as `feature_spec.json`)."""

    categorical: list[CategoricalField]
    vocabularies: dict[str, dict[str, int]]
    dense: list[str]
    dense_mean: list[float]
    dense_std: list[float]
    groups: list[Group]
    character_id_dropout: float = Field(default=0.1, ge=0, lt=1)

    def cardinalities(self) -> list[int]:
        """Vocabulary size per categorical field, including the OOV index 0."""
        return [len(self.vocabularies[f.name]) + 1 for f in self.categorical]

    def field_index(self, name: str) -> int:
        """Position of a categorical field in the encoded matrix."""
        return next(i for i, f in enumerate(self.categorical) if f.name == name)


class Encoded(BaseModel):
    """Model-ready arrays."""

    model_config = {"arbitrary_types_allowed": True}

    categorical: npt.NDArray[np.int64]
    dense: npt.NDArray[np.float32]


def fit_spec(train: pl.DataFrame, groups: set[Group], text_dims: int = 0) -> FeatureSpec:
    """Fit vocabularies and dense scaling on the training split, for the selected groups."""
    fields = [f for f in CATEGORICAL_FIELDS if f.group in groups]
    vocabularies: dict[str, dict[str, int]] = {}
    for field in fields:
        counts = train.group_by(field.name).len().filter(pl.col("len") >= field.min_count).sort(field.name)
        vocabularies[field.name] = {v: i + 1 for i, v in enumerate(counts[field.name].cast(pl.Utf8).to_list())}
    dense = [name for name, group in DENSE_FIELDS.items() if group in groups]
    if Group.TEXT in groups:
        dense += [f"{TEXT_PREFIX}{i}" for i in range(text_dims)]
    stats = train.select(dense).fill_null(0.0) if dense else pl.DataFrame()
    mean = [float(stats[c].mean() or 0.0) for c in dense]  # pyright: ignore[reportArgumentType]
    std = [max(float(stats[c].std() or 1.0), 1e-6) for c in dense]  # pyright: ignore[reportArgumentType]
    return FeatureSpec(
        categorical=fields,
        vocabularies=vocabularies,
        dense=dense,
        dense_mean=mean,
        dense_std=std,
        groups=sorted(groups),
    )


def encode(spec: FeatureSpec, frame: pl.DataFrame) -> Encoded:
    """Encode derived columns into integer ids (OOV = 0) and standardized dense values."""
    if frame.height <= SMALL_FRAME:
        # Serving path: N candidates. Dict lookups beat building polars hash maps per call (~10x).
        categorical = np.array(
            [
                [spec.vocabularies[f.name].get(v, 0) for v in frame[f.name].cast(pl.Utf8).to_list()]
                for f in spec.categorical
            ],
            dtype=np.int64,
        ).T.reshape(frame.height, len(spec.categorical))
    else:
        categorical = frame.select(
            pl.col(f.name).cast(pl.Utf8).replace_strict(spec.vocabularies[f.name], default=0, return_dtype=pl.Int64)
            for f in spec.categorical
        ).to_numpy()
    if spec.dense:
        raw = frame.select(pl.col(spec.dense).cast(pl.Float64).fill_null(0.0).fill_nan(0.0)).to_numpy()
        raw[~np.isfinite(raw)] = 0.0  # defensive: a malformed input must not become an infinite activation
        dense = ((raw - np.asarray(spec.dense_mean)) / np.asarray(spec.dense_std)).astype(np.float32)
    else:
        dense = np.zeros((frame.height, 0), dtype=np.float32)
    return Encoded(categorical=np.ascontiguousarray(categorical, dtype=np.int64), dense=np.ascontiguousarray(dense))
