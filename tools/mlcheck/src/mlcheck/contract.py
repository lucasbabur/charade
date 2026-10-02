"""Artifact contract: the files a training/evaluation run must write into `artifacts_dir`.

Tabular artifacts (parquet):
    splits.parquet       id (str), split (str), ts (datetime)
    predictions.parquet  id (str), split (str), model (str), label (int 0/1), pred (float), ts (datetime),
                         slice_<name> (str) for every required slice

JSON artifacts are the pydantic models below. `evaluation_ledger.jsonl` and `decisions.jsonl`
hold one model per line.

Evidence binding: the manifest and every evidence report (parity, latency, OPE) carry `bundle_sha256`,
the digest of the configured `bundle_files` (`bundle_digest`), so a report measured on another model
fails MLR005 instead of passing quietly.
"""

import hashlib
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


def bundle_digest(directory: Path, files: Iterable[str]) -> str:
    """sha256 over `name NUL sha256(file) LF` for each bundle file in sorted name order."""
    outer = hashlib.sha256()
    for name in sorted(files):
        inner = hashlib.sha256()
        with (directory / name).open("rb") as handle:
            while chunk := handle.read(1 << 20):
                inner.update(chunk)
        outer.update(f"{name}\0{inner.hexdigest()}\n".encode())
    return outer.hexdigest()


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Window(_Strict):
    """Inclusive time window of a split."""

    start: datetime
    end: datetime


class FittedArtifact(_Strict):
    """Anything fitted on data: vocabularies, encoders, priors, scalers, models, calibrators."""

    name: str
    fit_split: str
    fit_end: datetime


class RunManifest(_Strict):
    """Provenance of one training run (manifest.json)."""

    run_id: str = Field(min_length=1)
    created_at: datetime
    git_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    git_dirty: bool
    config_hash: str = Field(min_length=8)
    data_sha256: dict[str, str] = Field(min_length=1)
    windows: dict[str, Window]
    fitted_artifacts: list[FittedArtifact] = Field(min_length=1)
    model_seeds: dict[str, list[int]]
    library_versions: dict[str, str] = Field(min_length=1)
    bundle_sha256: str = Field(min_length=64)


class LeakageReport(_Strict):
    """Leakage probes (leakage.json)."""

    shuffled_label_auc: float
    feature_auc: dict[str, float]
    adversarial_auc: float


class ParityReport(_Strict):
    """Offline pipeline vs serving path vs exported model (parity.json)."""

    n_rows: int = Field(gt=0)
    train_serve_max_abs_diff: float
    categorical_mismatches: int
    onnx_max_abs_diff: float
    bundle_sha256: str


class LatencyReport(_Strict):
    """Load-test summary (latency.json)."""

    source: str
    n_candidates: int = Field(gt=0)
    requests: int = Field(gt=0)
    duration_s: float = Field(gt=0)
    p50_ms: float
    p95_ms: float
    p99_ms: float
    error_rate: float
    bundle_sha256: str
    """Digest the load-tested API reported at `GET /v1/model`."""


class PolicyEstimate(_Strict):
    """Off-policy estimate of one ranking policy."""

    name: str
    estimator: str
    value: float
    ci_low: float
    ci_high: float
    ess: float
    n: int
    max_weight: float


class OpeReport(_Strict):
    """Off-policy evaluation (ope.json)."""

    policies: list[PolicyEstimate] = Field(min_length=1)
    bundle_sha256: str


class DriftReport(_Strict):
    """PSI per feature per period against the training reference (drift.json)."""

    reference: str
    psi: dict[str, dict[str, float]] = Field(min_length=1)


class LedgerEntry(_Strict):
    """One evaluation event (evaluation_ledger.jsonl)."""

    timestamp: datetime
    run_id: str
    model_version: str
    split: str
    config_hash: str | None = None
    """Identity of everything that can be chosen while looking at results (model config, feature groups,
    policy, split windows). None only for entries written before this field existed."""
    window: str | None = None
    """The evaluated split's time window, e.g. "2014-10-29T00:00:00/2014-10-30T05:00:00"."""


class DecisionCandidate(_Strict):
    """One candidate as seen by the ranking policy."""

    candidate_id: str
    pctr: float
    gated: bool
    gate_reasons: list[str] = Field(default_factory=list[str])
    propensity: float | None = None
    """Probability that the policy serves this candidate for this request, when the policy logs it."""


class Decision(_Strict):
    """One logged ranking decision (decisions.jsonl)."""

    request_id: str
    candidates: list[DecisionCandidate]
    chosen_id: str | None
    propensity: float | None
    explored: bool
