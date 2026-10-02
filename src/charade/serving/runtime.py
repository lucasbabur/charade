"""Everything a worker loads once: scorer, character table, evidence, policy, store."""

import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from charade.config import PolicyConfig, Settings
from charade.ranking.evidence import Evidence
from charade.scoring.scorer import CHARACTERS_FILE, EVIDENCE_FILE, Scorer, bundle_sha256
from charade.serving.store import FeatureStore, MemoryStore, RedisStore


@dataclass
class Runtime:
    """Loaded bundle. `scorer is None` means not ready (no bundle at the configured path)."""

    scorer: Scorer | None
    characters: dict[str, dict[str, object]]
    evidence: Evidence
    policy: PolicyConfig
    store: FeatureStore
    model_version: str
    bundle_sha256: str = ""
    """Digest of the loaded bundle files (empty when none is loaded); evidence reports name it."""


def load_runtime(settings: Settings, store: FeatureStore | None = None) -> Runtime:
    """Load the bundle from `settings.artifacts_dir`; tolerate its absence (readiness reports it)."""
    directory: Path = settings.artifacts_dir
    if store is None:
        store = (
            RedisStore(settings.redis_url, settings.store_timeout_ms / 1000) if settings.redis_url else MemoryStore()
        )
    if not (directory / "model.onnx").is_file():
        return Runtime(None, {}, Evidence(counts={}), settings.policy, store, "none")
    characters = pl.read_parquet(directory / CHARACTERS_FILE)
    table = {
        row["character_id"]: row
        for row in characters.select(
            "character_id", "genre", "safety_tier", "creator_type", "num_interactions", "created_at"
        ).iter_rows(named=True)
    }
    manifest = directory / "manifest.json"
    version = json.loads(manifest.read_text())["run_id"] if manifest.is_file() else "unversioned"
    return Runtime(
        scorer=Scorer(directory),
        characters=table,
        evidence=Evidence.model_validate_json((directory / EVIDENCE_FILE).read_text()),
        policy=settings.policy,
        store=store,
        model_version=version,
        bundle_sha256=bundle_sha256(directory),
    )
