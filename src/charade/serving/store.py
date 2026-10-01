"""Online user-history store. Redis in production, in-memory for tests and local runs.

One key per user holding the `UserHistory` JSON (a few hundred bytes; per-campaign entries are
capped). Reads use one pipelined round trip with a hard timeout; on timeout or error the caller
gets `None` per user and serves with cold-user defaults (`degraded = true`), never an error.
"""

import asyncio
from collections.abc import Sequence
from typing import Protocol

from redis.asyncio import Redis
from redis.exceptions import RedisError

from charade.features.counters import UserHistory

KEY_PREFIX = "charade:user:"
TTL_SECONDS = 14 * 24 * 3600
MAX_CAMPAIGNS = 200


class StoreUnavailableError(Exception):
    """The store could not answer within its budget."""


class FeatureStore(Protocol):
    """What serving needs from the store."""

    async def get(self, user: str) -> UserHistory | None:
        """History for a user, or None if never seen. Raises StoreUnavailableError."""
        ...

    async def record(self, user: str, hour: int, campaign: str, clicked: bool) -> None:
        """Add an impression (and its click) to a user's history."""
        ...

    async def ping(self) -> bool:
        """True when the store is reachable."""
        ...


def _trim(history: UserHistory) -> UserHistory:
    if len(history.campaigns) > MAX_CAMPAIGNS:
        keep = sorted(history.campaigns.items(), key=lambda kv: kv[1][1], reverse=True)[:MAX_CAMPAIGNS]
        history.campaigns = dict(keep)
    return history


class MemoryStore:
    """Process-local store (tests, single-process local runs)."""

    def __init__(self) -> None:
        self.users: dict[str, UserHistory] = {}

    async def get(self, user: str) -> UserHistory | None:
        """History or None."""
        found = self.users.get(user)
        return None if found is None else found.model_copy(deep=True)

    async def record(self, user: str, hour: int, campaign: str, clicked: bool) -> None:
        """Update in place."""
        history = self.users.setdefault(user, UserHistory())
        history.record(hour, campaign, int(clicked))
        _trim(history)

    async def ping(self) -> bool:
        """Always reachable."""
        return True


class RedisStore:
    """Redis-backed store with a per-call timeout."""

    def __init__(self, url: str, timeout_s: float) -> None:
        self.redis = Redis.from_url(url, socket_timeout=timeout_s, socket_connect_timeout=timeout_s)
        self.timeout_s = timeout_s

    async def get(self, user: str) -> UserHistory | None:
        """One GET bounded by the timeout."""
        try:
            raw = await asyncio.wait_for(self.redis.get(KEY_PREFIX + user), self.timeout_s)
        except (TimeoutError, RedisError, OSError) as exc:
            raise StoreUnavailableError(str(exc)) from exc
        return None if raw is None else UserHistory.model_validate_json(raw)

    async def record(self, user: str, hour: int, campaign: str, clicked: bool) -> None:
        """Read-modify-write. Event ingestion is off the request path, so a lost race costs one count."""
        history = await self.get(user) or UserHistory()
        history.record(hour, campaign, int(clicked))
        try:
            await self.redis.set(KEY_PREFIX + user, _trim(history).model_dump_json(), ex=TTL_SECONDS)
        except (TimeoutError, RedisError, OSError) as exc:
            raise StoreUnavailableError(str(exc)) from exc

    async def ping(self) -> bool:
        """PING within the timeout."""
        try:
            return bool(await asyncio.wait_for(self.redis.ping(), self.timeout_s))  # pyright: ignore[reportUnknownArgumentType, reportGeneralTypeIssues]
        except (TimeoutError, RedisError, OSError):
            return False


async def get_many(store: FeatureStore, users: Sequence[str]) -> list[UserHistory | None]:
    """Convenience for batch reads."""
    return [await store.get(u) for u in users]
