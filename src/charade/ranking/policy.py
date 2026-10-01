"""Turn calibrated pCTRs for N candidates into a ranked, gated, explained decision.

Order of operations (docs/05-ranking-policy.md):
1. Hard gates: brand safety (advertiser x character tier), frequency cap, exhausted budget.
   Gated candidates are never served, whatever their score.
2. Value: pCTR x bid x pacing multiplier. With bid = 1 and no pacing this is pure CTR ranking.
3. Uncertainty: each candidate's pCTR becomes Beta(pCTR * n, (1 - pCTR) * n), where n is how much
   evidence training had for its (campaign, genre) pair, capped at `max_evidence`.
4. Decision: greedy on value for most traffic; Thompson sampling on an exploration bucket chosen
   by hashing the request id (deterministic and auditable). The propensity of the served ad is
   logged so every decision can feed off-policy evaluation and unbiased retraining.
"""

import hashlib
from enum import StrEnum

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field

type Floats = npt.NDArray[np.float64]


class GateReason(StrEnum):
    """Why a candidate cannot be served."""

    BRAND_SAFETY = "brand_safety"
    FREQUENCY_CAP = "frequency_cap"
    BUDGET_EXHAUSTED = "budget_exhausted"


TIER_ORDER = {"sfw": 0, "suggestive": 1, "mature": 2}


class PolicyConfig(BaseModel):
    """`[tool.charade.policy]`."""

    exploration_rate: float = Field(default=0.05, ge=0, le=1)
    min_evidence: float = 20.0
    max_evidence: float = 1000.0
    frequency_cap: int = 8
    """Max earlier impressions of one campaign to one user before the campaign is gated for that user."""
    propensity_draws: int = 64
    advertiser_max_tier: dict[str, str] = Field(default_factory=dict[str, str])
    """Advertiser (C21) -> highest character safety tier it accepts. Unlisted advertisers accept all tiers."""
    low_confidence_overlap: bool = True


class Candidate(BaseModel):
    """What the policy needs to know about one candidate ad."""

    candidate_id: str
    advertiser_id: str
    campaign_id: str
    pctr: float
    evidence: float = 0.0
    bid: float = 1.0
    pacing: float = Field(default=1.0, ge=0, le=1)
    prior_exposures: int = 0
    budget_exhausted: bool = False
    logit: float = 0.0
    """Raw model logit. Isotonic calibration is a step function, so equal pCTRs are common; the
    logit (strictly monotone in the model's belief) breaks those ties before the candidate id does."""


class Ranked(BaseModel):
    """One candidate in the response, in rank order (gated ones last)."""

    candidate_id: str
    rank: int | None
    pctr: float
    pctr_low: float
    pctr_high: float
    value: float
    gated: bool
    gate_reasons: list[GateReason]


class Decision(BaseModel):
    """Policy output."""

    ranked: list[Ranked]
    chosen_id: str | None
    propensity: float | None
    explored: bool
    confidence: str
    """`high`, or `low` when the top two candidates' 90 % intervals overlap."""


def gate_reasons(candidate: Candidate, character_tier: str, config: PolicyConfig) -> list[GateReason]:
    """All gates a candidate fails."""
    reasons: list[GateReason] = []
    max_tier = config.advertiser_max_tier.get(candidate.advertiser_id)
    if max_tier is not None and TIER_ORDER[character_tier] > TIER_ORDER[max_tier]:
        reasons.append(GateReason.BRAND_SAFETY)
    if candidate.prior_exposures >= config.frequency_cap:
        reasons.append(GateReason.FREQUENCY_CAP)
    if candidate.budget_exhausted:
        reasons.append(GateReason.BUDGET_EXHAUSTED)
    return reasons


def explores(request_id: str, rate: float) -> bool:
    """Deterministic exploration bucket: the same request id always lands in the same bucket."""
    bucket = int.from_bytes(hashlib.sha256(request_id.encode()).digest()[:8], "big") / 2**64
    return bucket < rate


def _rng(request_id: str) -> np.random.Generator:
    return np.random.default_rng(int.from_bytes(hashlib.sha256(b"ts:" + request_id.encode()).digest()[:8], "big"))


def _beta_params(pctr: Floats, evidence: Floats, config: PolicyConfig) -> tuple[Floats, Floats]:
    n = np.clip(evidence, config.min_evidence, config.max_evidence)
    p = np.clip(pctr, 1e-4, 1 - 1e-4)
    return p * n, (1 - p) * n


def decide(candidates: list[Candidate], character_tier: str, request_id: str, config: PolicyConfig) -> Decision:
    """Rank candidates and choose one (or none if every candidate is gated)."""
    reasons = [gate_reasons(c, character_tier, config) for c in candidates]
    open_idx = np.array([i for i, r in enumerate(reasons) if not r], dtype=np.int64)
    pctr = np.array([c.pctr for c in candidates], dtype=np.float64)
    weight = np.array([c.bid * c.pacing for c in candidates], dtype=np.float64)
    value = pctr * weight
    a, b = _beta_params(pctr, np.array([c.evidence for c in candidates], dtype=np.float64), config)
    low, high = _beta_quantiles(a, b)

    chosen: int | None = None
    propensity: float | None = None
    explored = False
    if len(open_idx):
        logits = np.array([candidates[i].logit for i in open_idx])
        ids = np.array([candidates[i].candidate_id for i in open_idx])
        greedy = int(open_idx[np.lexsort((ids, -logits, -value[open_idx]))[0]])
        draws = (
            _rng(request_id).beta(a[open_idx], b[open_idx], size=(config.propensity_draws, len(open_idx)))
            * weight[open_idx]
        )
        winners = open_idx[draws.argmax(axis=1)]
        ts_share = np.array([(winners == i).mean() for i in open_idx])
        explored = explores(request_id, config.exploration_rate)
        chosen = int(winners[0]) if explored else greedy
        position = int(np.flatnonzero(open_idx == chosen)[0])
        propensity = float(
            (1 - config.exploration_rate) * (chosen == greedy) + config.exploration_rate * ts_share[position]
        )

    order = sorted(
        range(len(candidates)),
        key=lambda i: (bool(reasons[i]), -value[i], -candidates[i].logit, candidates[i].candidate_id),
    )
    ranked = [
        Ranked(
            candidate_id=candidates[i].candidate_id,
            rank=None if reasons[i] else rank,
            pctr=float(pctr[i]),
            pctr_low=float(low[i]),
            pctr_high=float(high[i]),
            value=float(value[i]),
            gated=bool(reasons[i]),
            gate_reasons=reasons[i],
        )
        for rank, i in enumerate(order, start=1)
    ]
    return Decision(
        ranked=ranked,
        chosen_id=None if chosen is None else candidates[chosen].candidate_id,
        propensity=propensity,
        explored=explored,
        confidence=_confidence(ranked),
    )


def _beta_quantiles(a: Floats, b: Floats) -> tuple[Floats, Floats]:
    """5 % and 95 % quantiles of Beta(a, b), normal approximation (a, b >= ~2 here, clipped to [0, 1])."""
    mean = a / (a + b)
    sd = np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
    return np.clip(mean - 1.645 * sd, 0.0, 1.0), np.clip(mean + 1.645 * sd, 0.0, 1.0)


def _confidence(ranked: list[Ranked]) -> str:
    open_ranked = [r for r in ranked if not r.gated]
    if len(open_ranked) < 2:
        return "high"
    first, second = open_ranked[0], open_ranked[1]
    return "low" if second.pctr_high >= first.pctr_low else "high"
