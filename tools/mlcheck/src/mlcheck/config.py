"""Project configuration, read from `[tool.mlcheck]` in pyproject.toml or the top level of mlcheck.toml."""

import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ForeignKey(BaseModel):
    """A reference from the event table to an entity table, optionally with a creation timestamp."""

    model_config = ConfigDict(extra="forbid")

    column: str
    ref_path: Path
    ref_column: str
    ref_created_at: str | None = None
    ref_time_format: str = "%Y-%m-%d"


class DataConfig(BaseModel):
    """Raw event table contract."""

    model_config = ConfigDict(extra="forbid")

    path: Path
    id: str
    label: str
    time: str
    time_format: str
    period: str = "1d"
    required_columns: list[str] = Field(default_factory=list[str])
    categorical: list[str] = Field(default_factory=list[str])
    foreign_keys: list[ForeignKey] = Field(default_factory=list[ForeignKey])


class Thresholds(BaseModel):
    """Every numeric gate in one place; override any field in config."""

    model_config = ConfigDict(extra="forbid")

    max_null_fraction: float = 0.0
    dominant_value_share: float = 0.5
    volume_mad_k: float = 3.0
    shuffled_auc_low: float = 0.47
    shuffled_auc_high: float = 0.53
    max_univariate_auc: float = 0.9
    max_adversarial_auc: float = 0.8
    min_seeds: int = 3
    max_test_ne: float = 0.98
    calibration_low: float = 0.9
    calibration_high: float = 1.1
    max_ece: float = 0.02
    ece_bins: int = 15
    bootstrap_resamples: int = 1000
    confidence: float = 0.95
    min_slice_rows: int = 500
    parity_tolerance: float = 1e-6
    onnx_tolerance: float = 1e-5
    max_p99_ms: float = 50.0
    max_error_rate: float = 0.001
    min_ess: float = 1000.0
    max_psi: float = 0.25


class MlcheckConfig(BaseModel):
    """Top-level mlcheck configuration."""

    model_config = ConfigDict(extra="forbid")

    package: str | None = None
    source_root: Path = Path("src")
    serving_package: str | None = None
    features_package: str | None = None
    training_only_modules: list[str] = Field(
        default_factory=lambda: ["torch", "lightgbm", "optuna", "mlflow", "sklearn", "xgboost"]
    )
    artifacts_dir: Path = Path("artifacts/current")
    primary_model: str = "primary"
    baseline_model: str = "baseline"
    split_order: list[str] = Field(default_factory=lambda: ["train", "val", "test"])
    holdout_split: str = "test"
    required_slices: list[str] = Field(default_factory=list[str])
    data: DataConfig | None = None
    thresholds: Thresholds = Field(default_factory=Thresholds)


class ConfigNotFoundError(Exception):
    """Raised when neither mlcheck.toml nor a [tool.mlcheck] table exists."""


def load_config(root: Path) -> MlcheckConfig:
    """Load configuration for the project at `root`.

    Args:
        root: Project directory.

    Returns:
        Parsed configuration.

    Raises:
        ConfigNotFoundError: No configuration found.
    """
    standalone = root / "mlcheck.toml"
    if standalone.is_file():
        return MlcheckConfig.model_validate(tomllib.loads(standalone.read_text()))
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        table = tomllib.loads(pyproject.read_text()).get("tool", {}).get("mlcheck")
        if table is not None:
            return MlcheckConfig.model_validate(table)
    raise ConfigNotFoundError(f"no mlcheck.toml or [tool.mlcheck] in {root}")
