"""FastAPI application: `POST /v1/rank` scores, gates and ranks N candidates for one chat moment."""

import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Annotated

import numpy as np
import structlog
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from charade import __version__
from charade.config import get_settings
from charade.ranking.policy import Candidate, Ranked, decide
from charade.serving import metrics
from charade.serving.assemble import assemble, epoch_hour, unknown_character, user_key
from charade.serving.runtime import Runtime, load_runtime
from charade.serving.schemas import (
    ColdStart,
    Health,
    ImpressionEvent,
    ModelInfo,
    RankedAd,
    RankRequest,
    RankResponse,
)
from charade.serving.store import StoreUnavailableError

log = structlog.get_logger()


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
    frame, encoded = assemble(body, character, history, scorer.spec)
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
    candidates = [
        Candidate(
            candidate_id=c.candidate_id,
            advertiser_id=c.C21,
            campaign_id=c.C17,
            pctr=float(p),
            evidence=runtime.evidence.lookup(c.C17, genre),
            bid=c.bid,
            prior_exposures=int(e),
            logit=float(z),
        )
        for c, p, z, e, ok in zip(body.candidates, pctr, logits, frame["user_campaign_imps"], finite, strict=True)
        if ok
    ]
    decision = decide(candidates, str(character["safety_tier"]), body.request_id, runtime.policy)
    done = time.perf_counter()
    _observe(decision.ranked, decision.chosen_id, decision.explored, cold_character, history is None, len(candidates))
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
        candidates=len(candidates),
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
        cold_start=ColdStart(character=cold_character, user=history is None),
        warnings=warnings,
        model_version=runtime.model_version,
        ranked=[RankedAd.model_validate(r.model_dump()) for r in decision.ranked],
    )


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


async def record_impression(event: ImpressionEvent, runtime: Annotated[Runtime, Depends(_runtime)]) -> Response:
    """Update the user's counters after an impression (and click) is observed."""
    try:
        await runtime.store.record(
            user_key(event.device_id, event.device_ip, event.device_model),
            epoch_hour(event.hour),
            event.campaign_id,
            event.clicked,
        )
    except StoreUnavailableError as exc:
        raise HTTPException(status_code=503, detail="feature store unavailable") from exc
    return Response(status_code=204)


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
        metrics.LATENCY.labels(request.url.path).observe(time.perf_counter() - start)
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
        status_code=204,
        tags=["events"],
        summary="Record a served impression and its click",
    )
    api.add_api_route("/v1/model", model_info, methods=["GET"], tags=["ops"], summary="Loaded model")
    api.add_api_route("/health", health, methods=["GET"], tags=["ops"], summary="Liveness probe")
    api.add_api_route("/ready", ready, methods=["GET"], tags=["ops"], summary="Readiness probe")
    api.add_api_route(
        "/metrics", prometheus, methods=["GET"], tags=["ops"], summary="Prometheus metrics", include_in_schema=False
    )
    return api


app = create_app()
