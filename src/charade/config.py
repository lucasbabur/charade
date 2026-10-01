"""Settings: `[tool.charade]` in pyproject.toml, overridden by `CHARADE_*` environment variables and `.env`."""

from datetime import datetime
from functools import cache
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    PyprojectTomlConfigSettingsSource,
    SettingsConfigDict,
)


def _secret(name: str) -> SecretStr | None:
    return Field(default=None, validation_alias=AliasChoices(name))


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
