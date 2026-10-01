"""Settings: `[tool.charade]` in pyproject.toml, overridden by `CHARADE_*` environment variables and `.env`."""

from datetime import datetime
from functools import cache
from pathlib import Path

from pydantic import AliasChoices, BaseModel, Field, SecretStr
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    PyprojectTomlConfigSettingsSource,
    SettingsConfigDict,
)


def _secret(name: str) -> SecretStr | None:
    return Field(default=None, validation_alias=AliasChoices(name))


class PolicyConfig(BaseModel):
    """`[tool.charade.policy]`."""

    exploration_rate: float = Field(default=0.05, ge=0, le=1)
    min_evidence: float = 20.0
    max_evidence: float = 1000.0
    frequency_cap: int = 8
    """Max earlier impressions of one campaign to one user before the campaign is gated for that user."""
    exploration_sharpness: float = Field(default=2.0, ge=0)
    """Exploration samples q_i proportional to (upper evidence bound x bid x pacing)^k: k = 0 is uniform,
    larger k favours plausible winners; uncertain candidates have higher upper bounds."""
    advertiser_max_tier: dict[str, str] = Field(default_factory=dict[str, str])
    """Advertiser (C21) -> highest character safety tier it accepts. Unlisted advertisers accept all tiers."""


class DcnConfig(BaseModel):
    """DCN-v2 architecture and optimisation (values chosen by `uv run poe tune`, see reports/tuning/)."""

    embedding_dim: int = 16
    cross_layers: int = 3
    cross_rank: int = 64
    hidden: list[int] = Field(default_factory=lambda: [256, 128])
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 4096
    max_epochs: int = 6
    evals_per_epoch: int = 4
    patience: int = 4
    character_id_dropout: float = 0.1
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2])


class GbdtConfig(BaseModel):
    """LightGBM yardstick settings."""

    learning_rate: float = 0.03
    num_leaves: int = 127
    min_data_in_leaf: int = 500
    cat_smooth: float = 50.0
    cat_l2: float = 10.0
    feature_fraction: float = 0.8
    lambda_l2: float = 0.0
    max_rounds: int = 3000


class ModelConfig(BaseModel):
    """What the shipped model uses."""

    groups: list[str] = Field(
        default_factory=lambda: [
            "context",
            "device",
            "ad",
            "character_meta",
            "character_id",
            "conversation",
            "user_history",
        ]
    )
    text_provider: str | None = None
    dcn: DcnConfig = Field(default_factory=DcnConfig)
    gbdt: GbdtConfig = Field(default_factory=GbdtConfig)


class Settings(BaseSettings):
    """Project settings. Secrets are read only from the environment, never from pyproject.toml."""

    model_config = SettingsConfigDict(
        pyproject_toml_table_header=("tool", "charade"),
        pyproject_toml_depth=2,
        env_prefix="CHARADE_",
        env_file=".env",
        extra="ignore",
    )

    data_dir: Path = Path()
    artifacts_dir: Path = Path("artifacts/current")
    seed: int = 20141021
    train_end: datetime
    val_end: datetime
    test_end: datetime | None = None
    """Last holdout hour (inclusive). None: every hour after `val_end` (the take-home's fixed split)."""
    model: ModelConfig = Field(default_factory=ModelConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    redis_url: str | None = None
    """`CHARADE_REDIS_URL`; without it serving uses an in-process store."""
    store_timeout_ms: float = 10.0

    gemini_api_key: SecretStr | None = _secret("GEMINI_API_KEY")
    voyage_api_key: SecretStr | None = _secret("VOYAGE_API_KEY")
    openai_api_key: SecretStr | None = _secret("OPENAI_API_KEY")
    anthropic_api_key: SecretStr | None = _secret("ANTHROPIC_API_KEY")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Precedence: init kwargs > environment > .env > pyproject.toml."""
        return init_settings, env_settings, dotenv_settings, PyprojectTomlConfigSettingsSource(settings_cls)


@cache
def get_settings() -> Settings:
    """Process-wide settings; required fields come from pyproject.toml, which pyright cannot see."""
    return Settings()  # pyright: ignore[reportCallIssue]
