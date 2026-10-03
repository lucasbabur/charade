"""Online user-history store. Redis in production, in-memory for tests and local runs.

Four kinds of key:
    charade:dec:<request>      Served {user, hour, candidate, campaign, genre, pctr, model_version}, once, 48 h
    charade:user:<user>        UserHistory JSON (counters; per-campaign exposure totals, hourly detail for 48 h)
    charade:imp:<impression>   Impression {request_id, served fields, clicked}, kept 48 h
    charade:pairs:<version>    hash {<campaign>|<genre>:expected, ...:clicks}: the live correction's evidence
                               for one model version, kept 48 h after its last event. An impression adds the
                               served pCTR to `expected`, a click adds 1 to `clicks`, as atomic increments
                               queued in that event's transaction. The sums compare clicks with one model's
                               pCTR, so each bundle starts from empty, as the E012 replay that chose the prior
                               starts each day from empty. One hash per version makes a request's read one HMGET.

One canonical identity per event. A decision is immutable: re-recording the same request id with the
same content is a duplicate, with different content a conflict. An impression event names only its
impression and request ids; user, hour and campaign come from the stored decision, never from the
caller, so online history and the training rows rebuilt from logs describe the same impression.

Writes are idempotent and atomic:
- an impression id is recorded at most once (a retry is a no-op; a retry naming another request is a
  conflict);
- a click is attributed to its impression's user and hour, at most once, even if it arrives late;
- Redis updates run in WATCH/MULTI transactions and retry on conflict, so concurrent events for the
  same user cannot overwrite each other. Pair keys are not watched: every event of one campaign in one
  genre hits the same key, and additive sums need no read-modify-write, so a hot pair never forces a retry.

The frequency cap counts every exposure in the user's history, which lives until 14 days pass with no
event for that user (the key's TTL). Per-campaign totals are never trimmed: the cap would silently reset
for any campaign dropped. The value stays bounded by the campaigns one user is shown within that TTL.

If a user key has expired but its impression key has not, a late click marks the impression clicked and
is logged, but does not recreate a history holding a click without its impression.

Reads use one GET with a hard timeout; on timeout or error the caller serves cold-user defaults
(`degraded = true`), never an error.
"""

import asyncio
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict
from redis.asyncio import Redis
from redis.exceptions import RedisError, WatchError

from charade.data.events import CLICK_WINDOW_HOURS
from charade.features.counters import UserHistory
from charade.ranking.correction import PairState, pair_key

DECISION_PREFIX = "charade:dec:"
USER_PREFIX = "charade:user:"
IMPRESSION_PREFIX = "charade:imp:"
PAIRS_PREFIX = "charade:pairs:"
USER_TTL_SECONDS = 14 * 24 * 3600
IMPRESSION_TTL_SECONDS = CLICK_WINDOW_HOURS * 3600
"""Decisions, impressions and pair sums live as long as a click can still be attributed (the label maturity window)."""
MAX_RETRIES = 50


class StoreUnavailableError(Exception):
    """The store could not answer within its budget."""


class Outcome(StrEnum):
    """Result of an event write."""

    RECORDED = "recorded"
    DUPLICATE = "duplicate"
    CONFLICT = "conflict"
    UNKNOWN_DECISION = "unknown_decision"
    UNKNOWN_IMPRESSION = "unknown_impression"


class Served(BaseModel):
    """What one decision served: the identity every later event for that request is checked against."""

    model_config = ConfigDict(frozen=True)

    user: str
    hour: int
    candidate_id: str
    campaign: str
    genre: str | None = None
    pctr: float | None = None
    model_version: str | None = None
    """Character genre, the model's pCTR of the served ad and the model that scored it. All three feed the
    (campaign, genre) correction; when any is missing (history replays) the correction is left untouched."""

    def pair(self) -> tuple[str, str] | None:
        """(model version, pair key) this decision updates, if it carries what the correction needs."""
        if self.genre is None or self.pctr is None or self.model_version is None:
            return None
        return self.model_version, pair_key(self.campaign, self.genre)


class Impression(BaseModel):
    """A served impression, for click attribution and deduplication."""

    request_id: str
    served: Served
    clicked: bool = False


type Recorded = tuple[Outcome, Impression | None]
"""An impression write's outcome and the canonical impression (None when there is none)."""


class FeatureStore(Protocol):
    """What serving needs from the store."""

    async def get(self, user: str) -> UserHistory | None:
        """History for a user, or None if never seen. Raises StoreUnavailableError."""
        ...

    async def record_decision(self, request_id: str, served: Served) -> Outcome:
        """Store what a request served, once; a different decision under the same id is a conflict."""
        ...

    async def record_impression(self, impression_id: str, request_id: str) -> Recorded:
        """Count the served impression of a stored decision once."""
        ...

    async def record_click(self, impression_id: str) -> Outcome:
        """Count a click once, attributed to its impression's user and hour."""
        ...

    async def pairs(self, model_version: str, keys: list[str]) -> list[PairState]:
        """One model version's correction evidence per pair key (empty when unseen). Raises StoreUnavailableError."""
        ...

    async def ping(self) -> bool:
        """True when the store is reachable."""
        ...


def _apply_impression(history: UserHistory | None, served: Served) -> UserHistory:
    history = history or UserHistory()
    history.record(served.hour, served.campaign, 0)
    return history


def _existing(impression: Impression, request_id: str) -> Recorded:
    return (Outcome.DUPLICATE if impression.request_id == request_id else Outcome.CONFLICT), impression


class MemoryStore:
    """Process-local store (tests, single-process local runs). No awaits inside a write, so writes are atomic."""

    def __init__(self) -> None:
        self.decisions: dict[str, Served] = {}
        self.users: dict[str, UserHistory] = {}
        self.impressions: dict[str, Impression] = {}
        self.pair_states: dict[tuple[str, str], PairState] = {}

    async def get(self, user: str) -> UserHistory | None:
        """History or None."""
        found = self.users.get(user)
        return None if found is None else found.model_copy(deep=True)

    async def record_decision(self, request_id: str, served: Served) -> Outcome:
        """Write once; compare on retry."""
        if request_id not in self.decisions:
            self.decisions[request_id] = served
            return Outcome.RECORDED
        return Outcome.DUPLICATE if self.decisions[request_id] == served else Outcome.CONFLICT

    async def record_impression(self, impression_id: str, request_id: str) -> Recorded:
        """Count once per impression id, with the decision's identity."""
        if impression_id in self.impressions:
            return _existing(self.impressions[impression_id], request_id)
        served = self.decisions.get(request_id)
        if served is None:
            return Outcome.UNKNOWN_DECISION, None
        impression = self.impressions[impression_id] = Impression(request_id=request_id, served=served)
        self.users[served.user] = _apply_impression(self.users.get(served.user), served)
        if (pair := served.pair()) is not None:
            self.pair_states.setdefault(pair, PairState()).expected += served.pctr or 0.0
        return Outcome.RECORDED, impression

    async def record_click(self, impression_id: str) -> Outcome:
        """Count once per impression id."""
        impression = self.impressions.get(impression_id)
        if impression is None:
            return Outcome.UNKNOWN_IMPRESSION
        if impression.clicked:
            return Outcome.DUPLICATE
        impression.clicked = True
        if impression.served.user in self.users:
            self.users[impression.served.user].record_click(impression.served.hour)
        if (pair := impression.served.pair()) is not None:
            self.pair_states.setdefault(pair, PairState()).clicks += 1.0
        return Outcome.RECORDED

    async def pairs(self, model_version: str, keys: list[str]) -> list[PairState]:
        """Evidence per key."""
        return [self.pair_states.get((model_version, k), PairState()).model_copy() for k in keys]

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

    async def record_decision(self, request_id: str, served: Served) -> Outcome:
        """SET NX; on an existing key, compare (a stored decision never changes, so no transaction is needed)."""
        key, value = DECISION_PREFIX + request_id, served.model_dump_json()
        if await self._call(self.redis.set(key, value, nx=True, ex=IMPRESSION_TTL_SECONDS)):
            return Outcome.RECORDED
        stored = await self._call(self.redis.get(key))
        return (
            Outcome.DUPLICATE
            if stored is not None and Served.model_validate_json(stored) == served
            else Outcome.CONFLICT
        )

    async def record_impression(self, impression_id: str, request_id: str) -> Recorded:
        """Read the immutable decision; WATCH impression and user; write both, and add to the pair, atomically."""
        raw_dec = await self._call(self.redis.get(DECISION_PREFIX + request_id))
        imp_key = IMPRESSION_PREFIX + impression_id
        served = None if raw_dec is None else Served.model_validate_json(raw_dec)
        user_key = USER_PREFIX + (served.user if served else "")
        pair = served.pair() if served else None
        found: list[Impression] = []

        async def attempt(pipe: Any) -> Outcome:
            current = await pipe.get(imp_key)
            if current is not None:
                found.append(Impression.model_validate_json(current))
                return _existing(found[-1], request_id)[0]
            if served is None:
                return Outcome.UNKNOWN_DECISION
            raw = await pipe.get(user_key)
            history = _apply_impression(None if raw is None else UserHistory.model_validate_json(raw), served)
            impression = Impression(request_id=request_id, served=served)
            pipe.multi()
            pipe.set(imp_key, impression.model_dump_json(), ex=IMPRESSION_TTL_SECONDS)
            pipe.set(user_key, history.model_dump_json(), ex=USER_TTL_SECONDS)
            if pair:
                version, key = pair
                pipe.hincrbyfloat(PAIRS_PREFIX + version, f"{key}:expected", served.pctr or 0.0)
                pipe.expire(PAIRS_PREFIX + version, IMPRESSION_TTL_SECONDS)
            await pipe.execute()
            found.append(impression)
            return Outcome.RECORDED

        outcome = await self._transact([imp_key, user_key], attempt)
        return outcome, found[-1] if found else None

    async def record_click(self, impression_id: str) -> Outcome:
        """WATCH the impression and its user; mark clicked and add the click to the impression's hour."""
        imp_key = IMPRESSION_PREFIX + impression_id
        raw_imp = await self._call(self.redis.get(imp_key))
        if raw_imp is None:
            return Outcome.UNKNOWN_IMPRESSION
        served = Impression.model_validate_json(raw_imp).served
        user_key = USER_PREFIX + served.user
        pair = served.pair()

        async def attempt(pipe: Any) -> Outcome:
            current = await pipe.get(imp_key)
            if current is None:
                return Outcome.UNKNOWN_IMPRESSION
            impression = Impression.model_validate_json(current)
            if impression.clicked:
                return Outcome.DUPLICATE
            raw = await pipe.get(user_key)
            impression.clicked = True
            pipe.multi()
            pipe.set(imp_key, impression.model_dump_json(), keepttl=True)
            if raw is not None:
                history = UserHistory.model_validate_json(raw)
                history.record_click(impression.served.hour)
                pipe.set(user_key, history.model_dump_json(), ex=USER_TTL_SECONDS)
            if pair:
                version, key = pair
                pipe.hincrbyfloat(PAIRS_PREFIX + version, f"{key}:clicks", 1.0)
                pipe.expire(PAIRS_PREFIX + version, IMPRESSION_TTL_SECONDS)
            await pipe.execute()
            return Outcome.RECORDED

        return await self._transact([imp_key, user_key], attempt)

    async def pairs(self, model_version: str, keys: list[str]) -> list[PairState]:
        """One HMGET on the version's hash, bounded by the timeout."""
        if not keys:
            return []
        fields = [f"{k}:{total}" for k in keys for total in ("expected", "clicks")]
        raw = await self._call(self.redis.hmget(PAIRS_PREFIX + model_version, fields))
        return [
            PairState(expected=float(e or 0.0), clicks=float(c or 0.0))
            for e, c in zip(raw[::2], raw[1::2], strict=True)
        ]

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
