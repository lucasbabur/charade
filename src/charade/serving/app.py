"""FastAPI application."""

from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel

from charade import __version__


class Health(BaseModel):
    """Liveness response."""

    status: Literal["ok"] = "ok"
    version: str


def create_app() -> FastAPI:
    """Build the API. Routes are added here as each capability lands."""
    api = FastAPI(
        title="Charade",
        version=__version__,
        description="Contextual ad CTR prediction and candidate ranking for AI companion chats.",
    )

    @api.get("/health", summary="Liveness probe", tags=["ops"])
    async def health() -> Health:  # pyright: ignore[reportUnusedFunction]
        """Return 200 while the process is up."""
        return Health(version=__version__)

    return api


app = create_app()
