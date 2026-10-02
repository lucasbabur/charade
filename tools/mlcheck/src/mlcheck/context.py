"""Lazy, cached access to the project's config, data and artifacts."""

import json
from functools import cached_property
from pathlib import Path

import polars as pl
from pydantic import BaseModel

from mlcheck.config import DataConfig, MlcheckConfig
from mlcheck.contract import (
    Decision,
    DriftReport,
    LatencyReport,
    LeakageReport,
    OpeReport,
    ParityReport,
    RunManifest,
)


class ArtifactMissingError(Exception):
    """A check needs an input (run artifact or source tree) that does not exist."""


class NotConfiguredError(Exception):
    """A check needs a config section the project did not declare."""


class Context:
    """Everything a check may read. Each artifact is loaded once."""

    def __init__(self, root: Path, config: MlcheckConfig) -> None:
        self.root = root
        self.config = config
        self.thresholds = config.thresholds

    def artifact(self, name: str) -> Path:
        """Path to an artifact, failing if the run did not write it."""
        path = self.root / self.config.artifacts_dir / name
        if not path.is_file():
            raise ArtifactMissingError(f"missing artifact {path.relative_to(self.root)}")
        return path

    def _json[T: BaseModel](self, name: str, model: type[T]) -> T:
        return model.model_validate_json(self.artifact(name).read_text())

    def _jsonl[T: BaseModel](self, name: str, model: type[T]) -> list[T]:
        lines = self.artifact(name).read_text().splitlines()
        return [model.model_validate(json.loads(line)) for line in lines if line.strip()]

    @property
    def data_config(self) -> DataConfig:
        """Data section of the config."""
        if self.config.data is None:
            raise NotConfiguredError("[tool.mlcheck.data] not configured")
        return self.config.data

    @cached_property
    def events(self) -> pl.DataFrame:
        """Raw event table, every column as string, plus parsed `__ts`."""
        cfg = self.data_config
        frame = pl.read_csv(self.root / cfg.path, infer_schema=False)
        if cfg.time in frame.columns:
            raw, fmt = pl.col(cfg.time), cfg.time_format
            if "%H" in fmt and "%M" not in fmt:  # chrono rejects an hour without minutes
                raw, fmt = raw + "00", fmt + "%M"
            frame = frame.with_columns(raw.str.strptime(pl.Datetime, fmt, strict=False).alias("__ts"))
        return frame

    def reference_table(self, path: Path) -> pl.DataFrame:
        """Entity table referenced by a foreign key, every column as string."""
        return pl.read_csv(self.root / path, infer_schema=False)

    @cached_property
    def manifest(self) -> RunManifest:
        """manifest.json."""
        return self._json("manifest.json", RunManifest)

    @cached_property
    def splits(self) -> pl.DataFrame:
        """splits.parquet."""
        return pl.read_parquet(self.artifact("splits.parquet"))

    @cached_property
    def predictions(self) -> pl.DataFrame:
        """predictions.parquet."""
        return pl.read_parquet(self.artifact("predictions.parquet"))

    @cached_property
    def leakage(self) -> LeakageReport:
        """leakage.json."""
        return self._json("leakage.json", LeakageReport)

    @cached_property
    def parity(self) -> ParityReport:
        """parity.json."""
        return self._json("parity.json", ParityReport)

    @cached_property
    def latency(self) -> LatencyReport:
        """latency.json."""
        return self._json("latency.json", LatencyReport)

    @cached_property
    def ope(self) -> OpeReport:
        """ope.json."""
        return self._json("ope.json", OpeReport)

    @cached_property
    def drift(self) -> DriftReport:
        """drift.json."""
        return self._json("drift.json", DriftReport)

    @cached_property
    def decisions(self) -> list[Decision]:
        """decisions.jsonl."""
        return self._jsonl("decisions.jsonl", Decision)
