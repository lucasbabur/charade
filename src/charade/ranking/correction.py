"""Live (campaign, genre) correction to the model's pCTR: a Gamma-Poisson posterior on a multiplier.

For one campaign `c` shown next to characters of genre `g`, two sums accumulate from served impressions:
    expected  sum of the model's pCTR over those impressions (clicks the model expected)
    clicks    clicks observed on them
The correction `r` multiplies the model's pCTR for every future (c, g) candidate. Its prior has mean 1
(trust the model) and strength `prior` expected clicks; observing `expected` and `clicks` gives
    r ~ Gamma(prior + clicks, prior + expected):  mean (prior + clicks) / (prior + expected),
                                                   sd   sqrt(prior + clicks) / (prior + expected).
A pair has graduated when `expected >= prior`: the logs weigh at least as much as the prior, so the
observed rate, not the model, now decides the pair's level. With a half-life, both sums decay per hour so
the correction follows drift. `prior` was chosen on the validation day (E012); the policy consumes the
mean as a multiplier and the sd as the width of the candidate's interval.
"""

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel

type Floats = npt.NDArray[np.float64]


class PairState(BaseModel):
    """Accumulated evidence for one (campaign, genre) pair; the store persists exactly this."""

    expected: float = 0.0
    clicks: float = 0.0
    hour: int | None = None
    """Epoch hour of the last update; decay is applied lazily from here."""


def pair_key(campaign: str, genre: str) -> str:
    """Store key of a pair."""
    return f"{campaign}|{genre}"


def posterior(expected: Floats, clicks: Floats, prior: float) -> tuple[Floats, Floats]:
    """(mean, sd) of the correction multiplier for arrays of pair sums."""
    scale = prior + expected
    return (prior + clicks) / scale, np.sqrt(prior + clicks) / scale


def graduated(expected: Floats, prior: float) -> npt.NDArray[np.bool_]:
    """True where the logged evidence outweighs the prior."""
    return expected >= prior


def decay_factor(hours: int, half_life_hours: float | None) -> float:
    """Multiplier that `hours` of silence apply to both sums (1 without a half-life)."""
    if half_life_hours is None or hours <= 0:
        return 1.0
    return float(0.5 ** (hours / half_life_hours))


def decayed(state: PairState, hour: int, half_life_hours: float | None) -> PairState:
    """The pair's sums as of `hour`, decay applied for the hours since its last update."""
    if state.hour is None or hour <= state.hour:
        return state
    factor = decay_factor(hour - state.hour, half_life_hours)
    return PairState(expected=state.expected * factor, clicks=state.clicks * factor, hour=hour)


def observe(
    state: PairState, hour: int, expected: float = 0.0, clicks: float = 0.0, half_life_hours: float | None = None
) -> PairState:
    """Add one impression's expected clicks or one click at `hour`, decaying older evidence first."""
    current = decayed(state, hour, half_life_hours)
    return PairState(
        expected=current.expected + expected, clicks=current.clicks + clicks, hour=max(hour, current.hour or hour)
    )
