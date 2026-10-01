import asyncio
from collections.abc import Awaitable

import pytest
from fakeredis import FakeAsyncRedis

from charade.serving.store import Impression, MemoryStore, Outcome, RedisStore


def _stores() -> list[MemoryStore | RedisStore]:
    return [MemoryStore(), RedisStore("redis://unused", 1.0, client=FakeAsyncRedis())]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
@pytest.mark.parametrize("make", [0, 1], ids=["memory", "redis"])
async def test_impressions_and_clicks_are_idempotent(make: int) -> None:
    store = _stores()[make]
    imp = Impression(user="u", hour=100, campaign="c")
    assert await store.record_impression("i1", imp) is Outcome.RECORDED
    assert await store.record_impression("i1", imp) is Outcome.DUPLICATE
    assert await store.record_click("i1") is Outcome.RECORDED
    assert await store.record_click("i1") is Outcome.DUPLICATE
    assert await store.record_click("nope") is Outcome.UNKNOWN_IMPRESSION
    history = await store.get("u")
    assert history is not None
    assert (history.imps, history.clicks) == (1, 1)


@pytest.mark.anyio
@pytest.mark.parametrize("make", [0, 1], ids=["memory", "redis"])
async def test_late_click_counts_at_the_impression_hour(make: int) -> None:
    store = _stores()[make]
    await store.record_impression("early", Impression(user="u", hour=100, campaign="c"))
    await store.record_impression("later", Impression(user="u", hour=105, campaign="c"))
    await store.record_click("early")
    history = await store.get("u")
    assert history is not None
    assert history.snapshot(101, ["c"])["user_clicks"][0] == 1


@pytest.mark.anyio
async def test_concurrent_redis_writes_lose_nothing() -> None:
    """200 impressions, 100 clicks and 50 retried impressions for one user, 16 writers at a time:
    every event is counted exactly once (WATCH/MULTI retries resolve the conflicts)."""
    store = RedisStore("redis://unused", 1.0, client=FakeAsyncRedis())
    gate = asyncio.Semaphore(16)

    async def bounded(write: Awaitable[Outcome]) -> Outcome:
        async with gate:
            return await write

    first = [Impression(user="u", hour=100 + i % 5, campaign=f"c{i % 3}") for i in range(200)]
    await asyncio.gather(*(bounded(store.record_impression(f"i{i}", imp)) for i, imp in enumerate(first)))
    retried = Impression(user="u", hour=100, campaign="c0")
    outcomes = await asyncio.gather(
        *(bounded(store.record_click(f"i{i}")) for i in range(0, 200, 2)),
        *(bounded(store.record_impression(f"i{i}", retried)) for i in range(50)),
    )
    assert outcomes.count(Outcome.DUPLICATE) == 50
    history = await store.get("u")
    assert history is not None
    assert (history.imps, history.clicks) == (200, 100)
    assert sum(history.campaign_totals.values()) == 200
