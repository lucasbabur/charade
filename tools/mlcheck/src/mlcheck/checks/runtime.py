"""Serving, ranking-policy and drift checks."""

from mlcheck.context import Context
from mlcheck.registry import check
from mlcheck.result import Outcome, Severity, Stage, failed, passed


@check(
    "MLV001",
    "train-serve-parity",
    Stage.SERVING,
    "The serving path must produce the same features as the offline pipeline for the same rows; any difference is "
    "train/serve skew.",
)
def train_serve_parity(ctx: Context) -> Outcome:
    """Max abs dense difference within `parity_tolerance` and zero categorical mismatches."""
    report = ctx.parity
    message = (
        f"{report.n_rows:,} rows: max dense diff {report.train_serve_max_abs_diff:.2e}, "
        f"{report.categorical_mismatches} categorical mismatches"
    )
    ok = report.train_serve_max_abs_diff <= ctx.thresholds.parity_tolerance and report.categorical_mismatches == 0
    return passed(message) if ok else failed(message)


@check(
    "MLV002",
    "onnx-parity",
    Stage.SERVING,
    "The exported model is what serves traffic; it must reproduce the trained model's outputs.",
)
def onnx_parity(ctx: Context) -> Outcome:
    """Max abs difference between exported and trained model outputs within `onnx_tolerance`."""
    diff = ctx.parity.onnx_max_abs_diff
    message = f"max abs diff {diff:.2e}"
    return (
        passed(message)
        if diff <= ctx.thresholds.onnx_tolerance
        else failed(f"{message} > {ctx.thresholds.onnx_tolerance}")
    )


@check(
    "MLV003",
    "latency-p99",
    Stage.SERVING,
    "The ad slot waits for the ranker; p99 over budget means blank slots or a timed-out auction.",
)
def latency_p99(ctx: Context) -> Outcome:
    """Load-test p99 at most `max_p99_ms`."""
    report = ctx.latency
    message = (
        f"p50 {report.p50_ms:.1f} / p95 {report.p95_ms:.1f} / p99 {report.p99_ms:.1f} ms, "
        f"{report.requests / report.duration_s:.0f} rps, N={report.n_candidates} ({report.source})"
    )
    return passed(message) if report.p99_ms <= ctx.thresholds.max_p99_ms else failed(message)


@check("MLV004", "load-error-rate", Stage.SERVING, "Fast responses do not count if they are errors.")
def load_error_rate(ctx: Context) -> Outcome:
    """Load-test error rate at most `max_error_rate`."""
    rate = ctx.latency.error_rate
    message = f"error rate {rate:.4%}"
    return passed(message) if rate <= ctx.thresholds.max_error_rate else failed(message)


@check(
    "MLP001",
    "ope-effective-sample-size",
    Stage.POLICY,
    "An importance-weighted estimate resting on a few heavy weights is noise; effective sample size says how many "
    "rows really support it.",
    Severity.WARNING,
)
def ope_effective_sample_size(ctx: Context) -> Outcome:
    """Every off-policy estimate has ESS of at least `min_ess`."""
    rows = [f"{p.name}/{p.estimator}: ESS {p.ess:,.0f} of {p.n:,}" for p in ctx.ope.policies]
    weak = [p for p in ctx.ope.policies if p.ess < ctx.thresholds.min_ess]
    return (
        failed(f"{len(weak)} estimates below ESS {ctx.thresholds.min_ess:,.0f}", rows)
        if weak
        else passed("ESS ok", rows)
    )


@check(
    "MLP002",
    "ope-intervals",
    Stage.POLICY,
    "Policy comparisons need intervals; an estimate outside its own CI or a CI with no width is a reporting bug.",
)
def ope_intervals(ctx: Context) -> Outcome:
    """Every estimate lies inside a non-degenerate CI."""
    bad = [
        f"{p.name}/{p.estimator}: {p.value:.5f} [{p.ci_low:.5f}, {p.ci_high:.5f}]"
        for p in ctx.ope.policies
        if not (p.ci_low < p.ci_high and p.ci_low <= p.value <= p.ci_high)
    ]
    return failed("invalid intervals", bad) if bad else passed(f"{len(ctx.ope.policies)} estimates with valid CIs")


@check(
    "MLP003",
    "decisions-respect-gates",
    Stage.POLICY,
    "A gated candidate (brand safety, frequency cap, budget) must never be served, whatever its score.",
)
def decisions_respect_gates(ctx: Context) -> Outcome:
    """In logged decisions the chosen candidate exists and is never gated; all-gated decisions choose nothing."""
    problems: list[str] = []
    for decision in ctx.decisions:
        by_id = {c.candidate_id: c for c in decision.candidates}
        open_ids = [c.candidate_id for c in decision.candidates if not c.gated]
        if decision.chosen_id is None:
            if open_ids:
                problems.append(f"{decision.request_id}: nothing chosen with {len(open_ids)} eligible candidates")
        elif decision.chosen_id not in by_id:
            problems.append(f"{decision.request_id}: chosen {decision.chosen_id} not among candidates")
        elif by_id[decision.chosen_id].gated:
            problems.append(
                f"{decision.request_id}: served gated {decision.chosen_id} {by_id[decision.chosen_id].gate_reasons}"
            )
    if not ctx.decisions:
        return failed("decisions.jsonl is empty")
    return (
        failed(f"{len(problems)} gate violations", problems[:20])
        if problems
        else passed(f"{len(ctx.decisions):,} decisions respect gates")
    )


@check(
    "MLP004",
    "logged-propensities",
    Stage.POLICY,
    "Future off-policy evaluation and unbiased retraining need the probability of every served ad, in (0, 1].",
)
def logged_propensities(ctx: Context) -> Outcome:
    """Every served decision logs a propensity in (0, 1]; a single eligible candidate must have propensity 1."""
    problems: list[str] = []
    for decision in ctx.decisions:
        if decision.chosen_id is None:
            continue
        p = decision.propensity
        eligible = sum(not c.gated for c in decision.candidates)
        if p is None or not 0 < p <= 1:
            problems.append(f"{decision.request_id}: propensity {p}")
        elif eligible == 1 and p != 1:
            problems.append(f"{decision.request_id}: one eligible candidate but propensity {p}")
    served = sum(d.chosen_id is not None for d in ctx.decisions)
    return (
        failed(f"{len(problems)} bad propensities", problems[:20])
        if problems
        else passed(f"{served:,} served decisions logged")
    )


@check(
    "MLX001",
    "feature-drift-psi",
    Stage.DRIFT,
    "PSI above 0.25 marks a feature whose distribution moved enough to invalidate what the model learned about it.",
    Severity.WARNING,
)
def feature_drift_psi(ctx: Context) -> Outcome:
    """Every feature's PSI per period against the reference stays at most `max_psi`."""
    limit = ctx.thresholds.max_psi
    drifted = [
        f"{feature} @ {period}: PSI {value:.3f}"
        for feature, periods in sorted(ctx.drift.psi.items())
        for period, value in sorted(periods.items())
        if value > limit
    ]
    if drifted:
        return failed(f"{len(drifted)} feature-periods above PSI {limit} vs {ctx.drift.reference}", drifted)
    return passed(f"{len(ctx.drift.psi)} features within PSI {limit} vs {ctx.drift.reference}")
