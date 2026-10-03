"""FastAPI application: `POST /v1/rank` scores, gates and ranks N candidates for one chat moment."""

import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Annotated

import numpy as np
import structlog
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from charade import __version__
from charade.config import get_settings
from charade.features.counters import UserHistory
from charade.ranking.correction import PairState, graduated, pair_key, posterior
from charade.ranking.policy import Candidate, Ranked, decide
from charade.serving import metrics
from charade.serving.assemble import assemble, epoch_hour, unknown_character, user_key
from charade.serving.logging import configure_logging
from charade.serving.runtime import Runtime, load_runtime
from charade.serving.schemas import (
    ClickEvent,
    ColdStart,
    EventResult,
    Health,
    ImpressionEvent,
    ModelInfo,
    RankedAd,
    RankRequest,
    RankResponse,
)
from charade.serving.store import Outcome, Served, StoreUnavailableError

log = structlog.get_logger()
UNMATCHED_ROUTE = "unmatched"


def _runtime(request: Request) -> Runtime:
    return request.app.state.runtime


def _ready_runtime(runtime: Annotated[Runtime, Depends(_runtime)]) -> Runtime:
    if runtime.scorer is None:
        raise HTTPException(status_code=503, detail="model bundle not loaded")
    return runtime


def _dedupe(body: RankRequest, warnings: list[str]) -> RankRequest:
    seen: set[str] = set()
    unique = [c for c in body.candidates if not (c.candidate_id in seen or seen.add(c.candidate_id))]
    if len(unique) < len(body.candidates):
        warnings.append(f"dropped {len(body.candidates) - len(unique)} duplicate candidate ids")
    turn = body.conversation_turn
    if turn > body.session_msg_count:
        warnings.append("conversation_turn > session_msg_count; clamped")
        turn = body.session_msg_count
    return body.model_copy(update={"candidates": unique, "conversation_turn": turn})


async def rank(body: RankRequest, runtime: Annotated[Runtime, Depends(_ready_runtime)]) -> RankResponse:
    """Score every candidate, apply gates, rank, and choose one to serve."""
    start = time.perf_counter()
    warnings: list[str] = []
    body = _dedupe(body, warnings)
    scorer = runtime.scorer
    assert scorer is not None  # noqa: S101 - guaranteed by the dependency
    character = runtime.characters.get(body.character_id)
    cold_character = character is None
    if character is None:
        character = body.character.model_dump() if body.character else unknown_character(body.hour)
    user = user_key(body.device_id, body.device_ip, body.device_model)
    degraded = False
    try:
        history = await runtime.store.get(user)
    except StoreUnavailableError:
        history, degraded = None, True
        metrics.DEGRADED.inc()
    fetched = time.perf_counter()
    _, encoded = assemble(body, character, history, scorer.spec)
    assembled = time.perf_counter()
    logits, pctr = scorer.score(encoded)
    scored = time.perf_counter()
    finite = np.isfinite(pctr)
    if not finite.all():
        metrics.SCORE_ERRORS.inc(int((~finite).sum()))
        warnings.append(f"dropped {int((~finite).sum())} candidates with non-finite scores")
        if not finite.any():
            raise HTTPException(status_code=503, detail="no candidate could be scored")
    genre = str(character["genre"])
    # Frequency cap counts every recorded exposure (current hour included), unlike the causal model
    # feature. With the store down there is no exposure state: the cap fails open, flagged as degraded.
    campaigns = [c.C17 for c in body.candidates]
    cap_counts = (history or UserHistory()).exposures_so_far(campaigns)
    expected, mean, sd, degraded = await _correction(runtime, campaigns, genre, degraded)
    if degraded:
        warnings.append("feature store unavailable: frequency cap not enforced, decision not stored")
        metrics.CAP_UNENFORCED.inc()
    candidates = [
        Candidate(
            candidate_id=c.candidate_id,
            advertiser_id=c.C21,
            pctr=float(p),
            correction=float(m),
            correction_sd=float(w),
            bid=c.bid,
            # pacing / budget_exhausted keep their defaults: no budget feed exists (charade.ranking.pacing).
            prior_exposures=int(e),
            logit=float(z),
        )
        for c, p, z, e, m, w, ok in zip(body.candidates, pctr, logits, cap_counts, mean, sd, finite, strict=True)
        if ok
    ]
    decision = decide(candidates, str(character["safety_tier"]), body.request_id, runtime.policy)
    if decision.chosen_id is not None and not degraded:
        served_pctr = next(c.pctr for c in candidates if c.candidate_id == decision.chosen_id)
        degraded = await _store_decision(runtime, body, user, decision.chosen_id, genre, served_pctr, warnings)
    done = time.perf_counter()
    graduated_pairs = graduated(expected, runtime.policy.correction_prior)
    _observe(decision.ranked, decision.chosen_id, decision.explored, cold_character, history is None, len(candidates))
    metrics.GRADUATED.labels("graduated").inc(int(graduated_pairs.sum()))
    metrics.GRADUATED.labels("cold").inc(int((~graduated_pairs).sum()))
    for stage, seconds in (
        ("fetch", fetched - start),
        ("assemble", assembled - fetched),
        ("score", scored - assembled),
        ("policy", done - scored),
    ):
        metrics.STAGE.labels(stage).observe(seconds)
    log.info(
        "decision",
        request_id=body.request_id,
        chosen_id=decision.chosen_id,
        propensity=decision.propensity,
        explored=decision.explored,
        # Full candidate set: what off-policy evaluation and counterfactual training need later.
        candidates=[
            {
                "candidate_id": r.candidate_id,
                "pctr": r.pctr,
                "value": r.value,
                "propensity": r.propensity,
                "gate_reasons": [str(g) for g in r.gate_reasons],
            }
            for r in decision.ranked
        ],
        character_id=body.character_id,
        hour=body.hour.isoformat(),
        # Everything needed to rebuild a training row for whichever candidate gets served.
        context=body.model_dump(mode="json", exclude={"candidates", "character", "request_id", "hour"}),
        ads={c.candidate_id: c.model_dump(exclude={"candidate_id", "bid"}) for c in body.candidates},
        model_version=runtime.model_version,
        degraded=degraded,
    )
    return RankResponse(
        request_id=body.request_id,
        chosen_id=decision.chosen_id,
        propensity=decision.propensity,
        explored=decision.explored,
        confidence="low" if decision.confidence == "low" else "high",
        degraded=degraded,
        cold_start=ColdStart(character=cold_character, user=history is None, pairs_cold=int((~graduated_pairs).sum())),
        warnings=warnings,
        model_version=runtime.model_version,
        ranked=[RankedAd.model_validate(r.model_dump()) for r in decision.ranked],
    )


async def _correction(
    runtime: Runtime, campaigns: list[str], genre: str, degraded: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """(expected clicks, multiplier mean, multiplier sd, degraded) of each candidate's (campaign, genre) pair.

    Without the store every pair sits at its prior: multiplier 1 with the prior's width.
    """
    pairs = [PairState() for _ in campaigns]
    if not degraded:
        try:
            pairs = await runtime.store.pairs([pair_key(c, genre) for c in campaigns])
        except StoreUnavailableError:
            degraded = True
            metrics.DEGRADED.inc()
    expected = np.array([s.expected for s in pairs], dtype=np.float64)
    clicks = np.array([s.clicks for s in pairs], dtype=np.float64)
    mean, sd = posterior(expected, clicks, runtime.policy.correction_prior)
    return expected, mean, sd, degraded


async def _store_decision(
    runtime: Runtime, body: RankRequest, user: str, chosen: str, genre: str, pctr: float, warnings: list[str]
) -> bool:
    """Record what this request served, so impression events take their identity from it. True if degraded."""
    campaign = next(c.C17 for c in body.candidates if c.candidate_id == chosen)
    served = Served(
        user=user, hour=epoch_hour(body.hour), candidate_id=chosen, campaign=campaign, genre=genre, pctr=pctr
    )
    try:
        outcome = await runtime.store.record_decision(body.request_id, served)
    except StoreUnavailableError:
        warnings.append("feature store unavailable: decision not stored, its impression cannot be recorded")
        metrics.DEGRADED.inc()
        return True
    if outcome is Outcome.CONFLICT:
        raise HTTPException(status_code=409, detail="request_id already served a different decision; use a new id")
    return False


def _observe(
    ranked: list[Ranked], chosen: str | None, explored: bool, cold_character: bool, cold_user: bool, n: int
) -> None:
    metrics.CANDIDATES.observe(n)
    if cold_character:
        metrics.COLD.labels("character").inc()
    if cold_user:
        metrics.COLD.labels("user").inc()
    for r in ranked:
        for reason in r.gate_reasons:
            metrics.GATED.labels(str(reason)).inc()
    if chosen is None:
        metrics.NO_FILL.inc()
    else:
        metrics.PCTR.observe(next(r.pctr for r in ranked if r.candidate_id == chosen))
    if explored:
        metrics.EXPLORED.inc()


async def record_impression(event: ImpressionEvent, runtime: Annotated[Runtime, Depends(_runtime)]) -> EventResult:
    """Count the served impression of a ranked request once; user, campaign and hour come from its decision."""
    try:
        outcome, impression = await runtime.store.record_impression(event.impression_id, event.request_id)
    except StoreUnavailableError as exc:
        raise HTTPException(status_code=503, detail="feature store unavailable") from exc
    if outcome is Outcome.UNKNOWN_DECISION:
        raise HTTPException(status_code=404, detail="unknown or expired request_id")
    if outcome is Outcome.CONFLICT:
        raise HTTPException(status_code=409, detail="impression_id already recorded for another request")
    assert impression is not None  # noqa: S101 - recorded and duplicate outcomes carry the impression
    # Durable outcome record (log pipeline -> S3), emitted on duplicates too: if a previous attempt
    # committed the state but died before logging, the retry restores the record. Delivery is
    # at-least-once; charade.data.events deduplicates by impression_id.
    log.info(
        "impression",
        impression_id=event.impression_id,
        request_id=impression.request_id,
        candidate_id=impression.served.candidate_id,
        hour=datetime.fromtimestamp(impression.served.hour * 3600, UTC).replace(tzinfo=None).isoformat(),
    )
    return EventResult(outcome="recorded" if outcome is Outcome.RECORDED else "duplicate")


async def record_click(event: ClickEvent, runtime: Annotated[Runtime, Depends(_runtime)]) -> EventResult:
    """Attribute a click to its impression's user and hour (once per impression id)."""
    try:
        outcome = await runtime.store.record_click(event.impression_id)
    except StoreUnavailableError as exc:
        raise HTTPException(status_code=503, detail="feature store unavailable") from exc
    if outcome is Outcome.UNKNOWN_IMPRESSION:
        raise HTTPException(status_code=404, detail="unknown or expired impression_id")
    log.info("click", impression_id=event.impression_id)  # at-least-once, as for impressions
    return EventResult(outcome="recorded" if outcome is Outcome.RECORDED else "duplicate")


async def health() -> Health:
    """Liveness: the process is up."""
    return Health(version=__version__)


async def ready(runtime: Annotated[Runtime, Depends(_runtime)]) -> Health:
    """Readiness: a model bundle is loaded (a missing store degrades, it does not block)."""
    if runtime.scorer is None:
        raise HTTPException(status_code=503, detail="model bundle not loaded")
    return Health(version=__version__)


async def model_info(runtime: Annotated[Runtime, Depends(_ready_runtime)]) -> ModelInfo:
    """Loaded model version, feature groups and calibrator."""
    assert runtime.scorer is not None  # noqa: S101
    return ModelInfo(
        model_version=runtime.model_version,
        bundle_sha256=runtime.bundle_sha256,
        feature_groups=[str(g) for g in runtime.scorer.spec.groups],
        calibrator=str(runtime.scorer.calibrator.kind),
        characters=len(runtime.characters),
    )


async def prometheus() -> Response:
    """Prometheus exposition."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def create_app(runtime: Runtime | None = None) -> FastAPI:
    """Build the API; the bundle loads at startup unless a runtime is injected (tests)."""

    @asynccontextmanager
    async def lifespan(api: FastAPI) -> AsyncGenerator[None]:
        configure_logging()
        api.state.runtime = runtime or load_runtime(get_settings())
        yield

    api = FastAPI(
        title="Charade",
        version=__version__,
        description="Contextual ad CTR prediction and candidate ranking for AI companion chats.",
        lifespan=lifespan,
    )

    @api.middleware("http")
    async def timing(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        # Label by the matched route template, never the raw path: arbitrary URLs (scanners, typos) would
        # otherwise each create a new Prometheus series and grow memory without bound.
        route = request.scope.get("route")
        metrics.LATENCY.labels(getattr(route, "path", UNMATCHED_ROUTE)).observe(time.perf_counter() - start)
        return response

    api.add_api_route(
        "/v1/rank",
        rank,
        methods=["POST"],
        response_model=RankResponse,
        tags=["ranking"],
        summary="Rank candidate ads for one chat moment",
    )
    api.add_api_route(
        "/v1/events/impression",
        record_impression,
        methods=["POST"],
        response_model=EventResult,
        tags=["events"],
        summary="Record the served impression of a ranked request (idempotent on impression_id)",
    )
    api.add_api_route(
        "/v1/events/click",
        record_click,
        methods=["POST"],
        response_model=EventResult,
        tags=["events"],
        summary="Record a click on a served impression (idempotent; late clicks go to the impression hour)",
    )
    api.add_api_route("/v1/model", model_info, methods=["GET"], tags=["ops"], summary="Loaded model")
    api.add_api_route("/health", health, methods=["GET"], tags=["ops"], summary="Liveness probe")
    api.add_api_route("/ready", ready, methods=["GET"], tags=["ops"], summary="Readiness probe")
    api.add_api_route(
        "/metrics", prometheus, methods=["GET"], tags=["ops"], summary="Prometheus metrics", include_in_schema=False
    )
    return api


app = create_app()
