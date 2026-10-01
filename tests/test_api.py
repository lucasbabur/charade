from pathlib import Path

import httpx
import pytest

from charade import __version__
from charade.serving.app import app
from charade.serving.openapi import CONTRACT_PATH, render

ROOT = Path(__file__).parents[1]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_health_reports_version() -> None:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}


def test_committed_openapi_contract_is_current() -> None:
    committed = (ROOT / CONTRACT_PATH).read_text()
    assert committed == render(), "API changed: run `uv run poe openapi` and commit docs/api/openapi.json"
