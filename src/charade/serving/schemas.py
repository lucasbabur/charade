"""Request and response models of the public API (the OpenAPI contract is generated from these)."""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from charade.ranking.policy import GateReason

MAX_CANDIDATES = 500

Token = Annotated[str, Field(min_length=1, max_length=64)]


def parse_hour(value: object) -> object:
    """Accept ISO-8601 or the dataset's `YYMMDDHH`; truncate to the hour."""
    if isinstance(value, str) and len(value) == 8 and value.isdigit():
        return datetime.strptime(value, "%y%m%d%H")
    return value


class AdCandidate(BaseModel):
    """One ad from the retrieval layer: slot position plus creative/advertiser attributes."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: Token
    banner_pos: Token
    C14: Token = Field(description="Creative id")
    C15: Token = Field(description="Creative width")
    C16: Token = Field(description="Creative height")
    C17: Token = Field(description="Campaign id")
    C18: Token
    C19: Token
    C21: Token = Field(description="Advertiser id")
    bid: float = Field(default=1.0, gt=0, le=1e6)


class CharacterInfo(BaseModel):
    """Metadata for a character the bundle may not know yet (published after the last refresh)."""

    model_config = ConfigDict(extra="forbid")

    genre: Token
    safety_tier: Literal["sfw", "suggestive", "mature"]
    creator_type: Token = "community"
    num_interactions: int = Field(default=0, ge=0)
    created_at: datetime


class RankRequest(BaseModel):
    """An ad opportunity inside a chat, with the candidates to rank."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "request_id": "req-123",
                    "hour": "2014-10-29T14:00:00",
                    "character_id": "53baf355c4",
                    "conversation_turn": 3,
                    "session_msg_count": 12,
                    "site_id": "85f751fd",
                    "site_domain": "c4e18dd6",
                    "site_category": "50e219e0",
                    "app_id": "ecad2386",
                    "app_domain": "7801e8d9",
                    "app_category": "07d7df22",
                    "device_id": "a99f214a",
                    "device_ip": "c6c8e44e",
                    "device_model": "8a4875bd",
                    "device_type": "1",
                    "device_conn_type": "0",
                    "C1": "1005",
                    "C20": "-1",
                    "candidates": [
                        {
                            "candidate_id": "ad-1",
                            "banner_pos": "0",
                            "C14": "15706",
                            "C15": "320",
                            "C16": "50",
                            "C17": "1722",
                            "C18": "0",
                            "C19": "35",
                            "C21": "79",
                        }
                    ],
                }
            ]
        },
    )

    request_id: Token
    hour: datetime = Field(description="UTC hour of the impression (ISO-8601 or YYMMDDHH)")
    character_id: Token
    conversation_turn: int = Field(ge=1, le=100_000)
    session_msg_count: int = Field(ge=1, le=100_000)
    site_id: Token
    site_domain: Token
    site_category: Token
    app_id: Token
    app_domain: Token
    app_category: Token
    device_id: Token
    device_ip: Token
    device_model: Token
    device_type: Token
    device_conn_type: Token
    C1: Token
    C20: Token
    candidates: list[AdCandidate] = Field(min_length=1, max_length=MAX_CANDIDATES)
    character: CharacterInfo | None = Field(
        default=None, description="Used only when `character_id` is missing from the loaded character table"
    )

    _hour = field_validator("hour", mode="before")(parse_hour)


class RankedAd(BaseModel):
    """One candidate in rank order; gated candidates come last with `rank = null`."""

    candidate_id: str
    rank: int | None
    pctr: float = Field(description="Calibrated click probability")
    pctr_low: float = Field(
        description="Lower bound of a heuristic evidence interval (training support for the campaign x genre); "
        "not a calibrated posterior"
    )
    pctr_high: float = Field(description="Upper bound of the same heuristic evidence interval")
    value: float = Field(description="pCTR x bid x pacing; the ranking key")
    gated: bool
    gate_reasons: list[GateReason]
    propensity: float = Field(description="Exact probability that this request serves this candidate (0 if gated)")


class ColdStart(BaseModel):
    """Which entities the system has no history for."""

    character: bool
    user: bool


class RankResponse(BaseModel):
    """The decision and the full ranked list."""

    request_id: str
    chosen_id: str | None = Field(description="Candidate to serve; null when every candidate is gated")
    propensity: float | None = Field(
        description="Probability the policy served `chosen_id` (log it with the impression)"
    )
    explored: bool
    confidence: Literal["high", "low"] = Field(
        description="Heuristic: low when the top two evidence intervals overlap; not a statistical guarantee"
    )
    degraded: bool = Field(description="True when the feature store was unavailable and defaults were used")
    cold_start: ColdStart
    warnings: list[str]
    model_version: str
    ranked: list[RankedAd]


class ImpressionEvent(BaseModel):
    """The ad a RankRequest chose was shown. Idempotent on `impression_id` (retries are no-ops).

    User, campaign, candidate and hour come from the stored decision, never from the event.
    """

    model_config = ConfigDict(extra="forbid")

    impression_id: Token = Field(description="Unique per served ad")
    request_id: Token = Field(description="The RankRequest whose chosen ad was shown")


class ClickEvent(BaseModel):
    """A click on a served impression, possibly hours later. Idempotent on `impression_id`."""

    model_config = ConfigDict(extra="forbid")

    impression_id: Token


class EventResult(BaseModel):
    """What the store did with the event."""

    outcome: Literal["recorded", "duplicate"]


class Health(BaseModel):
    """Liveness response."""

    status: Literal["ok"] = "ok"
    version: str


class ModelInfo(BaseModel):
    """What is loaded."""

    model_version: str
    bundle_sha256: str = Field(description="Digest of the loaded bundle; load-test and parity reports must name it")
    feature_groups: list[str]
    calibrator: str
    characters: int
