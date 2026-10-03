"""Turn calibrated pCTRs for N candidates into a ranked, gated, explained decision.

Order of operations (docs/04-ranking-policy.md):
1. Hard gates: brand safety (advertiser x character tier), frequency cap, exhausted budget.
   Gated candidates are never served, whatever their score.
2. Value: pCTR x bid x pacing multiplier. With bid = 1 and no pacing this is pure CTR ranking.
3. Uncertainty: each pCTR is multiplied by the live (campaign, genre) correction (`charade.ranking.correction`)
   and gets the interval pCTR x (correction +- 1.645 sd). The width reflects how many outcomes the pair has
   logged; it is not a calibrated posterior over prediction error.
4. Decision, with exact selection probabilities: on a hashed exploration bucket of rate eps the ad
   is sampled from q_i proportional to (upper interval bound x bid x pacing)^k over eligible ads;
   otherwise the greedy ad is served. Every eligible ad therefore has the closed-form probability
       p_i = (1 - eps) * 1[i is greedy] + eps * q_i
   which is logged for every candidate. (An earlier Thompson-sampling version estimated p_i from
   the same draws that picked the ad, which biased it upward; a recovery test now guards this.)
"""

import hashlib
from enum import StrEnum

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field

from charade.config import PolicyConfig

type Floats = npt.NDArray[np.float64]


class GateReason(StrEnum):
    """Why a candidate cannot be served."""

    BRAND_SAFETY = "brand_safety"
    FREQUENCY_CAP = "frequency_cap"
    BUDGET_EXHAUSTED = "budget_exhausted"


TIER_ORDER = {"sfw": 0, "suggestive": 1, "mature": 2}


class Candidate(BaseModel):
    """What the policy needs to know about one candidate ad."""

    candidate_id: str
    advertiser_id: str
    pctr: float
    correction: float = 1.0
    """Posterior mean of the pair's pCTR multiplier (1 with no logged outcomes)."""
    correction_sd: float = 0.0
    """Posterior sd of that multiplier; 0 means no interval."""
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
    correction: float
    pctr_low: float
    pctr_high: float
    value: float
    gated: bool
    gate_reasons: list[GateReason]
    propensity: float
    """Exact probability that this policy serves this candidate for this request (0 if gated)."""


class Decision(BaseModel):
    """Policy output."""

    ranked: list[Ranked]
    chosen_id: str | None
    propensity: float | None
    explored: bool
    confidence: str
    """Heuristic: `low` when the top two candidates' correction intervals overlap, else `high`. Not a
    calibrated probability statement."""


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


def _uniform(request_id: str) -> float:
    """Second hash, independent of the bucket hash: the exploration draw for this request."""
    return int.from_bytes(hashlib.sha256(b"explore:" + request_id.encode()).digest()[:8], "big") / 2**64


def selection_probabilities(
    order: list[int], upper_value: Floats, eligible: npt.NDArray[np.bool_], config: PolicyConfig
) -> tuple[Floats, int | None, Floats]:
    """(p over all candidates, greedy index, exploration distribution q). Gated candidates get 0.

    `order` is the full ranking (best first); greedy is its first eligible candidate.
    """
    n = len(upper_value)
    p, q = np.zeros(n), np.zeros(n)
    if not eligible.any():
        return p, None, q
    idx = np.flatnonzero(eligible)
    greedy = next(i for i in order if eligible[i])
    weights = np.clip(upper_value[idx], 1e-12, None) ** config.exploration_sharpness
    q[idx] = weights / weights.sum()
    p = config.exploration_rate * q
    p[greedy] += 1 - config.exploration_rate
    return p, greedy, q


def decide(candidates: list[Candidate], character_tier: str, request_id: str, config: PolicyConfig) -> Decision:
    """Rank candidates and choose one (or none if every candidate is gated)."""
    reasons = [gate_reasons(c, character_tier, config) for c in candidates]
    pctr = np.array([c.pctr for c in candidates], dtype=np.float64)
    weight = np.array([c.bid * c.pacing for c in candidates], dtype=np.float64)
    mean = np.array([c.correction for c in candidates], dtype=np.float64)
    sd = np.array([c.correction_sd for c in candidates], dtype=np.float64)
    value = np.clip(pctr * mean, 0.0, 1.0) * weight
    low, high = _interval(pctr, mean, sd)
    eligible = np.array([not r for r in reasons], dtype=bool)
    # Rank key: value, then raw logit (isotonic calibration ties pCTRs), then candidate id.
    order = sorted(range(len(candidates)), key=lambda i: (-value[i], -candidates[i].logit, candidates[i].candidate_id))
    probabilities, greedy, q = selection_probabilities(order, high * weight, eligible, config)

    chosen: int | None = None
    explored = False
    if greedy is not None:
        explored = explores(request_id, config.exploration_rate)
        if explored:
            idx = np.flatnonzero(eligible)
            cumulative = np.cumsum(q[idx])
            position = int(np.searchsorted(cumulative, _uniform(request_id) * cumulative[-1], side="right"))
            chosen = int(idx[min(position, len(idx) - 1)])
        else:
            chosen = greedy

    order = [i for i in order if not reasons[i]] + [i for i in order if reasons[i]]  # gated last
    ranked = [
        Ranked(
            candidate_id=candidates[i].candidate_id,
            rank=None if reasons[i] else rank,
            pctr=float(pctr[i]),
            correction=float(mean[i]),
            pctr_low=float(low[i]),
            pctr_high=float(high[i]),
            value=float(value[i]),
            gated=bool(reasons[i]),
            gate_reasons=reasons[i],
            propensity=float(probabilities[i]),
        )
        for rank, i in enumerate(order, start=1)
    ]
    return Decision(
        ranked=ranked,
        chosen_id=None if chosen is None else candidates[chosen].candidate_id,
        propensity=None if chosen is None else float(probabilities[chosen]),
        explored=explored,
        confidence=_confidence(ranked),
    )


def _interval(pctr: Floats, mean: Floats, sd: Floats) -> tuple[Floats, Floats]:
    """5 % and 95 % bounds of pCTR x correction (normal approximation of the Gamma posterior), clipped to [0, 1]."""
    return np.clip(pctr * (mean - 1.645 * sd), 0.0, 1.0), np.clip(pctr * (mean + 1.645 * sd), 0.0, 1.0)


def _confidence(ranked: list[Ranked]) -> str:
    open_ranked = [r for r in ranked if not r.gated]
    if len(open_ranked) < 2:
        return "high"
    first, second = open_ranked[0], open_ranked[1]
    return "low" if second.pctr_high >= first.pctr_low else "high"
