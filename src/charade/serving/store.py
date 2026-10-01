"""Online user-history store. Redis in production, in-memory for tests and local runs.

Two kinds of key:
    charade:user:<user>        UserHistory JSON (counters; per-campaign detail capped)
    charade:imp:<impression>   {user, hour, campaign, clicked}, kept 48 h

Writes are idempotent and atomic:
- an impression id is recorded at most once (a retried event is a no-op);
- a click is attributed to its impression's user and hour, at most once, even if it arrives late;
- Redis updates run in WATCH/MULTI transactions and retry on conflict, so concurrent events for the
  same user cannot overwrite each other.

Reads use one GET with a hard timeout; on timeout or error the caller serves cold-user defaults
(`degraded = true`), never an error.
"""

import asyncio
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel
from redis.asyncio import Redis
from redis.exceptions import RedisError, WatchError

from charade.features.counters import UserHistory

USER_PREFIX = "charade:user:"
IMPRESSION_PREFIX = "charade:imp:"
USER_TTL_SECONDS = 14 * 24 * 3600
IMPRESSION_TTL_SECONDS = 48 * 3600
MAX_CAMPAIGNS = 200
MAX_RETRIES = 50


class StoreUnavailableError(Exception):
    """The store could not answer within its budget."""


class Outcome(StrEnum):
    """Result of an event write."""

    RECORDED = "recorded"
    DUPLICATE = "duplicate"
    UNKNOWN_IMPRESSION = "unknown_impression"


class Impression(BaseModel):
    """What the store remembers about a served impression, for click attribution and deduplication."""

    user: str
    hour: int
    campaign: str
    clicked: bool = False


class FeatureStore(Protocol):
    """What serving needs from the store."""

    async def get(self, user: str) -> UserHistory | None:
        """History for a user, or None if never seen. Raises StoreUnavailableError."""
        ...

    async def record_impression(self, impression_id: str, impression: Impression) -> Outcome:
        """Count a served impression once."""
        ...

    async def record_click(self, impression_id: str) -> Outcome:
        """Count a click once, attributed to its impression's user and hour."""
        ...

    async def ping(self) -> bool:
        """True when the store is reachable."""
        ...


def _trim(history: UserHistory) -> UserHistory:
    """Keep per-campaign detail for the most recently active campaigns only (bounded value size)."""
    if len(history.campaign_totals) > MAX_CAMPAIGNS:
        recency = {c: max(h.keys(), default=-1) for c, h in history.campaign_buckets.items()}
        keep = set(sorted(history.campaign_totals, key=lambda c: recency.get(c, -1), reverse=True)[:MAX_CAMPAIGNS])
        history.campaign_totals = {c: n for c, n in history.campaign_totals.items() if c in keep}
        history.campaign_buckets = {c: b for c, b in history.campaign_buckets.items() if c in keep}
    return history


def _apply_impression(history: UserHistory | None, impression: Impression) -> UserHistory:
    history = history or UserHistory()
    history.record(impression.hour, impression.campaign, 0)
    return _trim(history)


class MemoryStore:
    """Process-local store (tests, single-process local runs). No awaits inside a write, so writes are atomic."""

    def __init__(self) -> None:
        self.users: dict[str, UserHistory] = {}
        self.impressions: dict[str, Impression] = {}

    async def get(self, user: str) -> UserHistory | None:
        """History or None."""
        found = self.users.get(user)
        return None if found is None else found.model_copy(deep=True)

    async def record_impression(self, impression_id: str, impression: Impression) -> Outcome:
        """Count once per impression id."""
        if impression_id in self.impressions:
            return Outcome.DUPLICATE
        self.impressions[impression_id] = impression
        self.users[impression.user] = _apply_impression(self.users.get(impression.user), impression)
        return Outcome.RECORDED

    async def record_click(self, impression_id: str) -> Outcome:
        """Count once per impression id."""
        impression = self.impressions.get(impression_id)
        if impression is None:
            return Outcome.UNKNOWN_IMPRESSION
        if impression.clicked:
            return Outcome.DUPLICATE
        impression.clicked = True
        self.users[impression.user].record_click(impression.hour)
        return Outcome.RECORDED

    async def ping(self) -> bool:
        """Always reachable."""
        return True


class RedisStore:
    """Redis-backed store: bounded reads, transactional idempotent writes."""

    def __init__(self, url: str, timeout_s: float, client: Redis | None = None) -> None:
        self.redis = client or Redis.from_url(url, socket_timeout=timeout_s, socket_connect_timeout=timeout_s)
        self.timeout_s = timeout_s

    async def get(self, user: str) -> UserHistory | None:
        """One GET bounded by the timeout."""
        try:
            raw = await asyncio.wait_for(self.redis.get(USER_PREFIX + user), self.timeout_s)
        except (TimeoutError, RedisError, OSError) as exc:
            raise StoreUnavailableError(str(exc)) from exc
        return None if raw is None else UserHistory.model_validate_json(raw)

    async def record_impression(self, impression_id: str, impression: Impression) -> Outcome:
        """WATCH both keys; skip if the impression exists; else write impression and history atomically."""
        imp_key, user_key = IMPRESSION_PREFIX + impression_id, USER_PREFIX + impression.user

        async def attempt(pipe: Any) -> Outcome:
            if await pipe.exists(imp_key):
                return Outcome.DUPLICATE
            raw = await pipe.get(user_key)
            history = _apply_impression(None if raw is None else UserHistory.model_validate_json(raw), impression)
            pipe.multi()
            pipe.set(imp_key, impression.model_dump_json(), ex=IMPRESSION_TTL_SECONDS)
            pipe.set(user_key, history.model_dump_json(), ex=USER_TTL_SECONDS)
            await pipe.execute()
            return Outcome.RECORDED

        return await self._transact([imp_key, user_key], attempt)

    async def record_click(self, impression_id: str) -> Outcome:
        """WATCH the impression and its user; mark clicked and add the click to the impression's hour."""
        imp_key = IMPRESSION_PREFIX + impression_id
        raw_imp = await self._call(self.redis.get(imp_key))
        if raw_imp is None:
            return Outcome.UNKNOWN_IMPRESSION
        user_key = USER_PREFIX + Impression.model_validate_json(raw_imp).user

        async def attempt(pipe: Any) -> Outcome:
            current = await pipe.get(imp_key)
            if current is None:
                return Outcome.UNKNOWN_IMPRESSION
            impression = Impression.model_validate_json(current)
            if impression.clicked:
                return Outcome.DUPLICATE
            raw = await pipe.get(user_key)
            history = UserHistory() if raw is None else UserHistory.model_validate_json(raw)
            history.record_click(impression.hour)
            impression.clicked = True
            pipe.multi()
            pipe.set(imp_key, impression.model_dump_json(), keepttl=True)
            pipe.set(user_key, history.model_dump_json(), ex=USER_TTL_SECONDS)
            await pipe.execute()
            return Outcome.RECORDED

        return await self._transact([imp_key, user_key], attempt)

    async def _transact(self, keys: list[str], attempt: Callable[[Any], Awaitable[Outcome]]) -> Outcome:
        try:
            async with self.redis.pipeline(transaction=True) as pipe:
                for attempt_no in range(MAX_RETRIES):
                    try:
                        await pipe.watch(*keys)
                        return await attempt(pipe)
                    except WatchError:
                        await pipe.reset()
                        await asyncio.sleep(0.0005 * attempt_no)  # back off so hot keys converge
                        continue
        except (TimeoutError, RedisError, OSError) as exc:
            raise StoreUnavailableError(str(exc)) from exc
        raise StoreUnavailableError(f"write conflict persisted after {MAX_RETRIES} retries")

    async def _call(self, awaitable: Awaitable[Any]) -> Any:
        try:
            return await asyncio.wait_for(awaitable, self.timeout_s)
        except (TimeoutError, RedisError, OSError) as exc:
            raise StoreUnavailableError(str(exc)) from exc

    async def ping(self) -> bool:
        """PING within the timeout."""
        try:
            return bool(await asyncio.wait_for(self.redis.ping(), self.timeout_s))  # pyright: ignore[reportUnknownArgumentType, reportGeneralTypeIssues]
        except (TimeoutError, RedisError, OSError):
            return False
