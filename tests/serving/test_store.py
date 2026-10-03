import asyncio
from collections.abc import Awaitable
from datetime import UTC, datetime

import pytest
from fakeredis import FakeAsyncRedis

from charade.serving.assemble import epoch_hour
from charade.serving.store import MemoryStore, Outcome, RedisStore, Served


def _stores() -> list[MemoryStore | RedisStore]:
    return [MemoryStore(), RedisStore("redis://unused", 1.0, client=FakeAsyncRedis())]


async def _serve(store: MemoryStore | RedisStore, impression_id: str, hour: int = 100, campaign: str = "c") -> Outcome:
    """Rank (store the decision, request id = impression id) then record its impression, for user "u"."""
    await store.record_decision(impression_id, Served(user="u", hour=hour, candidate_id="a", campaign=campaign))
    return (await store.record_impression(impression_id, impression_id))[0]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
@pytest.mark.parametrize("make", [0, 1], ids=["memory", "redis"])
async def test_impressions_and_clicks_are_idempotent(make: int) -> None:
    store = _stores()[make]
    assert await _serve(store, "i1") is Outcome.RECORDED
    assert (await store.record_impression("i1", "i1"))[0] is Outcome.DUPLICATE
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
    await _serve(store, "early", hour=100)
    await _serve(store, "later", hour=105)
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

    async def retry(impression_id: str) -> Outcome:
        return (await store.record_impression(impression_id, impression_id))[0]

    await asyncio.gather(*(bounded(_serve(store, f"i{i}", 100 + i % 5, f"c{i % 3}")) for i in range(200)))
    outcomes = await asyncio.gather(
        *(bounded(store.record_click(f"i{i}")) for i in range(0, 200, 2)),
        *(bounded(retry(f"i{i}")) for i in range(50)),
    )
    assert outcomes.count(Outcome.DUPLICATE) == 50
    history = await store.get("u")
    assert history is not None
    assert (history.imps, history.clicks) == (200, 100)
    assert sum(history.campaign_totals.values()) == 200


@pytest.mark.anyio
@pytest.mark.parametrize("make", [0, 1], ids=["memory", "redis"])
async def test_cap_totals_survive_many_newer_campaigns(make: int) -> None:
    """Eight exposures of one campaign stay counted after 250 newer campaigns (trimming once reset them to 0)."""
    store = _stores()[make]
    for i in range(8):
        await _serve(store, f"old-{i}", campaign="capped")
    for i in range(250):
        await _serve(store, f"new-{i}", hour=101, campaign=f"c{i}")
    history = await store.get("u")
    assert history is not None
    assert history.exposures_so_far(["capped"]).tolist() == [8.0]


@pytest.mark.anyio
async def test_a_click_after_the_user_key_expired_does_not_invent_a_history() -> None:
    client = FakeAsyncRedis()
    store = RedisStore("redis://unused", 1.0, client=client)
    await _serve(store, "i1")
    await client.delete("charade:user:u")
    assert await store.record_click("i1") is Outcome.RECORDED
    assert await store.get("u") is None
    assert await store.record_click("i1") is Outcome.DUPLICATE


def test_epoch_hour_treats_naive_datetimes_as_utc() -> None:
    naive, aware = datetime(2014, 10, 29, 10), datetime(2014, 10, 29, 10, tzinfo=UTC)
    assert epoch_hour(naive) == epoch_hour(aware) == int(aware.timestamp()) // 3600


@pytest.mark.anyio
@pytest.mark.parametrize("make", [0, 1], ids=["memory", "redis"])
async def test_decisions_are_immutable_and_impressions_take_their_identity(make: int) -> None:
    store = _stores()[make]
    served = Served(user="u", hour=100, candidate_id="a", campaign="c")
    assert await store.record_decision("r", served) is Outcome.RECORDED
    assert await store.record_decision("r", served) is Outcome.DUPLICATE
    assert await store.record_decision("r", served.model_copy(update={"user": "v"})) is Outcome.CONFLICT
    assert (await store.record_impression("i", "unknown"))[0] is Outcome.UNKNOWN_DECISION
    outcome, impression = await store.record_impression("i", "r")
    assert (outcome, impression and impression.served) == (Outcome.RECORDED, served)
    assert (await store.record_impression("i", "other"))[0] is Outcome.CONFLICT
    assert await store.get("v") is None


@pytest.mark.anyio
@pytest.mark.parametrize("make", [0, 1], ids=["memory", "redis"])
async def test_impressions_and_clicks_feed_the_pair_correction_once(make: int) -> None:
    """Served pCTR goes to `expected`, a click to `clicks`; retries change nothing; unseen pairs are empty."""
    store = _stores()[make]
    served = Served(user="u", hour=100, candidate_id="a", campaign="camp", genre="romance", pctr=0.25)
    await store.record_decision("r", served)
    assert (await store.record_impression("i", "r"))[0] is Outcome.RECORDED
    assert (await store.record_impression("i", "r"))[0] is Outcome.DUPLICATE
    assert await store.record_click("i") is Outcome.RECORDED
    assert await store.record_click("i") is Outcome.DUPLICATE
    state, other = await store.pairs(["camp|romance", "camp|horror"])
    assert (state.expected, state.clicks) == (pytest.approx(0.25), 1.0)
    assert (other.expected, other.clicks) == (0.0, 0.0)


@pytest.mark.anyio
@pytest.mark.parametrize("make", [0, 1], ids=["memory", "redis"])
async def test_decisions_without_genre_or_pctr_leave_the_correction_alone(make: int) -> None:
    store = _stores()[make]
    await _serve(store, "i1", campaign="camp")
    await store.record_click("i1")
    (state,) = await store.pairs(["camp|romance"])
    assert (state.expected, state.clicks) == (0.0, 0.0)


@pytest.mark.anyio
async def test_concurrent_pair_updates_lose_nothing() -> None:
    """Same (campaign, genre) hammered by 16 writers: expected and clicks equal the event counts."""
    store = RedisStore("redis://unused", 1.0, client=FakeAsyncRedis())
    gate = asyncio.Semaphore(16)

    async def serve(i: int) -> Outcome:
        async with gate:
            served = Served(user=f"u{i}", hour=100, candidate_id="a", campaign="camp", genre="g", pctr=0.5)
            await store.record_decision(f"r{i}", served)
            return (await store.record_impression(f"i{i}", f"r{i}"))[0]

    async def click(i: int) -> Outcome:
        async with gate:
            return await store.record_click(f"i{i}")

    await asyncio.gather(*(serve(i) for i in range(100)))
    await asyncio.gather(*(click(i) for i in range(0, 100, 2)))
    (state,) = await store.pairs(["camp|g"])
    assert state.expected == pytest.approx(50.0)
    assert state.clicks == 50.0
