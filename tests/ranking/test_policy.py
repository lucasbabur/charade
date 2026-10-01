import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from charade.ranking.pacing import Pacer
from charade.ranking.policy import Candidate, GateReason, PolicyConfig, decide, explores

CONFIG = PolicyConfig(advertiser_max_tier={"family": "sfw"})


def _candidates(pctrs: list[float], **overrides: object) -> list[Candidate]:
    base = {"advertiser_id": "adv", "campaign_id": "camp", "evidence": 500.0}
    return [Candidate(candidate_id=f"c{i}", pctr=p, **(base | overrides)) for i, p in enumerate(pctrs)]  # type: ignore[arg-type]


def test_greedy_picks_highest_value_with_propensity_near_one() -> None:
    decision = decide(_candidates([0.1, 0.3, 0.2]), "sfw", "req-not-exploring", PolicyConfig(exploration_rate=0.0))
    assert decision.chosen_id == "c1"
    assert decision.propensity == 1.0
    assert [r.candidate_id for r in decision.ranked] == ["c1", "c2", "c0"]


def test_brand_safety_gates_mature_characters_for_family_advertisers() -> None:
    candidates = [*_candidates([0.9], advertiser_id="family"), *_candidates([0.1])]
    decision = decide(candidates, "mature", "r", CONFIG)
    assert decision.ranked[-1].gate_reasons == [GateReason.BRAND_SAFETY]
    assert decision.chosen_id == "c0"
    assert decision.ranked[0].candidate_id == "c0"
    assert decide(candidates, "sfw", "r", CONFIG).ranked[0].pctr == 0.9


def test_frequency_cap_and_budget_gate() -> None:
    capped = _candidates([0.5], prior_exposures=CONFIG.frequency_cap)
    broke = _candidates([0.5], budget_exhausted=True)
    assert decide(capped, "sfw", "r", CONFIG).ranked[0].gate_reasons == [GateReason.FREQUENCY_CAP]
    decision = decide(broke, "sfw", "r", CONFIG)
    assert decision.chosen_id is None
    assert decision.propensity is None


def test_exploration_bucket_is_deterministic_and_near_rate() -> None:
    hits = sum(explores(f"req-{i}", 0.05) for i in range(20_000))
    assert 0.04 < hits / 20_000 < 0.06
    assert explores("abc", 0.05) == explores("abc", 0.05)


def test_uncertain_close_candidates_are_flagged_low_confidence() -> None:
    thin = _candidates([0.20, 0.19], evidence=20.0)
    thick = _candidates([0.30, 0.10], evidence=1000.0)
    assert decide(thin, "sfw", "r", CONFIG).confidence == "low"
    assert decide(thick, "sfw", "r", CONFIG).confidence == "high"


def test_pacer_throttles_overspend_and_stops_at_budget() -> None:
    pacer = Pacer(daily_budget=100)
    assert pacer.update(0.5, 80) < 0.5
    fresh = Pacer(daily_budget=100)
    assert fresh.update(0.5, 20) == 1.0
    assert fresh.update(0.6, 90) == 0.0
    assert fresh.exhausted


pctrs = st.lists(st.floats(0.001, 0.999), min_size=1, max_size=12)


@settings(max_examples=200, deadline=None)
@given(pctrs, st.lists(st.booleans(), min_size=12, max_size=12), st.text(min_size=1, max_size=12))
def test_policy_invariants(values: list[float], family: list[bool], request_id: str) -> None:
    candidates = [
        Candidate(
            candidate_id=f"c{i}", advertiser_id="family" if family[i] else "other", campaign_id="x", pctr=p, evidence=50
        )
        for i, p in enumerate(values)
    ]
    decision = decide(candidates, "mature", request_id, CONFIG)
    by_id = {r.candidate_id: r for r in decision.ranked}
    assert len(decision.ranked) == len(candidates)
    if decision.chosen_id is None:
        assert all(r.gated for r in decision.ranked)
    else:
        assert not by_id[decision.chosen_id].gated
        assert decision.propensity is not None
        assert 0 < decision.propensity <= 1
    open_values = [r.value for r in decision.ranked if not r.gated]
    assert open_values == sorted(open_values, reverse=True)
    assert all(r.pctr_low <= r.pctr_high for r in decision.ranked)
    again = decide(candidates, "mature", request_id, CONFIG)
    assert again == decision
    assert np.isfinite([r.value for r in decision.ranked]).all()
