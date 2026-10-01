from datetime import datetime

import pytest

from charade.config import get_settings


@pytest.fixture(autouse=True)
def _fresh_settings() -> None:
    get_settings.cache_clear()


def test_reads_tool_charade_table_from_pyproject() -> None:
    settings = get_settings()
    assert settings.train_end == datetime(2014, 10, 27, 23)
    assert settings.train_end < settings.val_end


def test_environment_overrides_pyproject(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHARADE_SEED", "7")
    assert get_settings().seed == 7


def test_api_keys_come_from_unprefixed_env_and_stay_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    settings = get_settings()
    assert settings.anthropic_api_key is not None
    assert settings.anthropic_api_key.get_secret_value() == "sk-test"
    assert "sk-test" not in repr(settings)
