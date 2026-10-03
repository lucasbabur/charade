"""Live (campaign, genre) correction to the model's pCTR: a Gamma-Poisson posterior on a multiplier.

For one campaign `c` shown next to characters of genre `g`, two sums accumulate from served impressions:
    expected  sum of the model's pCTR over those impressions (clicks the model expected)
    clicks    clicks observed on them
The correction `r` multiplies the model's pCTR for every future (c, g) candidate. Its prior has mean 1
(trust the model) and strength `prior` expected clicks; observing `expected` and `clicks` gives
    r ~ Gamma(prior + clicks, prior + expected):  mean (prior + clicks) / (prior + expected),
                                                   sd   sqrt(prior + clicks) / (prior + expected).
A pair has graduated when `expected >= prior`: the logs weigh at least as much as the prior, so the
observed rate, not the model, now decides the pair's level. `prior` was chosen on the validation day (E012);
the policy consumes the mean as a multiplier and the sd as the width of the candidate's interval. The sums
are plain additive counters, so the store updates them with atomic increments and no transaction. A decay
was replayed in E012 and was indistinguishable from none on this data, so none is applied.
"""

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel

type Floats = npt.NDArray[np.float64]


class PairState(BaseModel):
    """Accumulated evidence for one (campaign, genre) pair; the store persists exactly this."""

    expected: float = 0.0
    clicks: float = 0.0


def pair_key(model_version: str, campaign: str, genre: str) -> str:
    """Store key of a pair, per model version.

    The sums compare clicks with one model's pCTR, so each bundle starts its own from empty, as the E012
    replay that chose `prior` starts each day from empty.
    """
    return f"{model_version}:{campaign}|{genre}"


def posterior(expected: Floats, clicks: Floats, prior: float) -> tuple[Floats, Floats]:
    """(mean, sd) of the correction multiplier for arrays of pair sums."""
    scale = prior + expected
    return (prior + clicks) / scale, np.sqrt(prior + clicks) / scale


def graduated(expected: Floats, prior: float) -> npt.NDArray[np.bool_]:
    """True where the logged evidence outweighs the prior."""
    return expected >= prior
