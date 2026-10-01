"""Offline evaluation of ranking policies on the test days (`uv run poe ope`).

The logs show one ad per impression, so a candidate set has to be reconstructed. For every
publisher x hour cell (`site_id`, `app_id`, `ts`), the candidates are the creatives served in that
cell (`C14` + `banner_pos`, with the creative's other ad fields), capped at the `TOP_K` most
frequent. The logging propensity of an ad is its share of the cell's impressions. Each evaluated
impression keeps its own context and user, and every candidate is scored as if it had been shown
there (including its own user x campaign exposure count). Single-creative cells are excluded (a
property of the cell, not of the logged action) and the coverage is reported.

The estimand is the CTR over every impression in a multi-creative cell. An impression whose logged ad
is outside the cell's top-K stays in that population: the evaluated policies never choose that ad, so
their probability of the logged action is 0, and `mu` stays the ad's share of the whole cell. (Dropping
those rows while keeping unconditional shares, as an earlier version did, conditions on the logged
action and biases IPS upward.)

Policies: logging (observed CTR), uniform random, greedy pCTR without gates, and the shipped policy.
The shipped policy's per-candidate probabilities, gates and ordering all come from the serving code
(`charade.ranking.policy.decide`); nothing about the policy is reimplemented here. Writes `ope.json`
and `decisions.jsonl` (a sample of those decisions, with exact propensities).

These are estimates under two assumptions that the bootstrap does not cover: candidate sets equal
what was served in a publisher x hour cell, and logging propensities equal impression shares.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from charade.config import Settings, get_settings
from charade.evaluation.ope import Estimate, estimate, lift
from charade.features.derive import derive
from charade.features.spec import Group, encode, fit_spec
from charade.models.dataset import build_frame
from charade.models.gbdt import predict_gbdt, train_gbdt
from charade.ranking.evidence import Evidence
from charade.ranking.policy import Candidate, Decision, decide, gate_reasons
from charade.scoring.scorer import EVIDENCE_FILE, Scorer

TOP_K = 10
AD_FIELDS = ("C14", "banner_pos", "C15", "C16", "C17", "C18", "C19", "C21")
CELL = ("site_id", "app_id", "ts")
DECISION_SAMPLE = 3000
REPORT = Path("reports/policy/ope.md")


def candidate_sets(test: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(top-K candidates per cell with `mu` and `slot`, evaluated rows with the logged slot or -1)."""
    shares = (
        test.group_by(*CELL, "C14", "banner_pos")
        .agg(pl.len().alias("count"), *[pl.col(f).mode().sort().first() for f in AD_FIELDS[2:]])
        .with_columns((pl.col("count") / pl.col("count").sum().over(CELL)).alias("mu"))
        .sort([*CELL, "count", "C14", "banner_pos"], descending=[False, False, False, True, False, False])
        .with_columns(pl.int_range(pl.len()).over(CELL).alias("slot"), pl.len().over(CELL).alias("k"))
        .filter(pl.col("k") >= 2)
    )
    rows = test.join(
        shares.select(*CELL, "C14", "banner_pos", "slot", "mu"), on=[*CELL, "C14", "banner_pos"], how="inner"
    ).with_columns(pl.when(pl.col("slot") < TOP_K).then(pl.col("slot")).otherwise(-1).alias("slot"))
    served = shares.filter(pl.col("slot") < TOP_K).with_columns(pl.len().over(CELL).alias("k"))
    return served, rows.rename({"slot": "logged_slot", "mu": "logged_mu"})


def _campaign_exposure(frame: pl.DataFrame) -> pl.DataFrame:
    """Cumulative impressions per (user, C17) including each hour, for 'strictly before hour h' lookups."""
    return (
        frame.group_by("user", "C17", "ts")
        .len()
        .sort("user", "C17", "ts")
        .with_columns(pl.col("len").cum_sum().over("user", "C17").alias("cum"))
        .drop("len")
    )


def expand(rows: pl.DataFrame, served: pl.DataFrame, history: pl.DataFrame) -> pl.DataFrame:
    """One row per (evaluated impression, candidate) with candidate ad fields and exposure counts."""
    context = rows.drop(*AD_FIELDS, "user_campaign_imps").with_row_index("row")
    pairs = context.join(served.select(*CELL, *AD_FIELDS, "slot", "mu"), on=list(CELL))
    pairs = pairs.with_columns((pl.col("ts") - timedelta(hours=1)).alias("before")).sort("user", "C17", "before")
    pairs = pairs.join_asof(
        history.rename({"ts": "before"}).sort("user", "C17", "before"),
        on="before",
        by=["user", "C17"],
        strategy="backward",
        check_sortedness=False,
    ).with_columns(pl.col("cum").fill_null(0).alias("user_campaign_imps"))
    return derive(pairs.drop("before", "cum")).sort("row", "slot")


def _matrix(pairs: pl.DataFrame, column: str, n_rows: int, fill: float) -> np.ndarray:
    out = np.full((n_rows, TOP_K), fill, dtype=np.float64)
    out[pairs["row"].to_numpy(), pairs["slot"].to_numpy()] = pairs[column].cast(pl.Float64).to_numpy()
    return out


def _row_candidates(pairs: pl.DataFrame) -> dict[int, list[tuple[int, Candidate, str]]]:
    """Per evaluated row: (slot, Candidate, character tier), built exactly as the API builds them."""
    out: dict[int, list[tuple[int, Candidate, str]]] = {}
    columns = (
        "row",
        "slot",
        "C14",
        "banner_pos",
        "C21",
        "C17",
        "pctr",
        "logit",
        "evidence",
        "user_campaign_imps",
        "safety_tier",
    )
    for row, slot, c14, pos, c21, c17, pctr, logit, evidence, exposures, tier in pairs.select(columns).iter_rows():
        candidate = Candidate(
            candidate_id=f"{c14}@{pos}",
            advertiser_id=c21,
            campaign_id=c17,
            pctr=pctr,
            logit=logit,
            evidence=evidence,
            prior_exposures=int(exposures),
        )
        out.setdefault(row, []).append((slot, candidate, tier))
    return out


def policies(rows: dict[int, list[tuple[int, Candidate, str]]], n: int, settings: Settings) -> dict[str, np.ndarray]:
    """pi(slot | row) for every policy, shape [rows, TOP_K].

    The shipped policy's probabilities come from `charade.ranking.policy.decide` itself (the
    serving code), so gates, ordering and exploration cannot drift from production.
    """
    uniform, greedy, shipped = (np.zeros((n, TOP_K)) for _ in range(3))
    for row, entries in rows.items():
        slots = [slot for slot, _, _ in entries]
        uniform[row, slots] = 1 / len(slots)
        best = min(entries, key=lambda e: (-e[1].pctr, -e[1].logit, e[1].candidate_id))
        greedy[row, best[0]] = 1.0
        decision = decide([c for _, c, _ in entries], entries[0][2], f"ope-{row}", settings.policy)
        by_id = {r.candidate_id: r.propensity for r in decision.ranked}
        for slot, candidate, _ in entries:
            shipped[row, slot] = by_id[candidate.candidate_id]
    return {
        "uniform random": uniform,
        "greedy pCTR (no gates)": greedy,
        "shipped policy (gates + 5% exploration)": shipped,
    }


@dataclass
class CandidateMatrices:
    """Reconstructed candidate sets on the test days, as [rows, TOP_K] matrices (unused slots masked)."""

    rows: pl.DataFrame
    pairs: pl.DataFrame
    test_rows: int
    served: pl.DataFrame
    candidates: dict[int, list[tuple[int, Candidate, str]]]
    mask: np.ndarray
    pctr: np.ndarray
    reward: np.ndarray
    """Independent outcome model's click probability (DR direct-method term)."""
    logit: np.ndarray
    gated: np.ndarray
    evidence: np.ndarray
    campaign: np.ndarray
    logged: np.ndarray
    y: np.ndarray
    mu: np.ndarray
    blocks: np.ndarray

    def at_logged(self, matrix: np.ndarray) -> np.ndarray:
        """Each row's entry for its logged ad; 0 when that ad is outside the candidate set."""
        inside = self.logged >= 0
        return np.where(inside, matrix[np.arange(len(self.y)), np.where(inside, self.logged, 0)], 0.0)


def reward_model(frame: pl.DataFrame, settings: Settings) -> Callable[[pl.DataFrame], np.ndarray]:
    """Independent outcome model for the doubly robust estimator.

    DR's direct-method term must not come from the policy being evaluated, or the estimate grades
    the model by its own beliefs. This is a LightGBM on the same features, trained on days before
    the last training day and early-stopped on that day, so it never sees validation or test.
    """
    train = frame.filter(pl.col("split") == "train")
    stamps: list[datetime] = train["ts"].to_list()
    cutoff = max(stamps).replace(hour=0)
    fit, stop = train.filter(pl.col("ts") < cutoff), train.filter(pl.col("ts") >= cutoff)
    spec = fit_spec(fit, {Group(g) for g in settings.model.groups})
    model = train_gbdt(
        spec,
        encode(spec, fit),
        fit["click"].to_numpy(),
        encode(spec, stop),
        stop["click"].to_numpy(),
        settings.model.gbdt,
        settings.seed,
    )
    return lambda rows: predict_gbdt(model, encode(spec, rows))


def load_candidates(settings: Settings, data_dir: Path | None = None, split: str = "test") -> CandidateMatrices:
    """Build candidate sets on `split` and score every (impression, candidate) pair.

    The shipped bundle scores for the policy; an independent reward model scores for DR.
    """
    art = settings.artifacts_dir
    scorer = Scorer(art)
    evidence = Evidence.model_validate_json((art / EVIDENCE_FILE).read_text())
    frame = build_frame(settings, data_dir or settings.data_dir)
    history = _campaign_exposure(frame)
    test = frame.filter(pl.col("split") == split)
    served, rows = candidate_sets(test)
    pairs = expand(rows, served, history)
    logits, pctrs = scorer.score(encode(scorer.spec, pairs))
    pairs = pairs.with_columns(
        pl.Series("pctr", pctrs), pl.Series("logit", logits), pl.Series("reward", reward_model(frame, settings)(pairs))
    )
    pairs = pairs.with_columns(
        pl.struct("C17", "genre")
        .map_elements(lambda r: evidence.lookup(r["C17"], r["genre"]), return_dtype=pl.Float64)
        .alias("evidence"),
    )
    candidates = _row_candidates(pairs)
    pairs = pairs.with_columns(
        pl.Series(
            "gated",
            [bool(gate_reasons(c, tier, settings.policy)) for _, c, tier in _in_pair_order(candidates, pairs)],
        )
    )
    n = rows.height
    ordered = rows.with_row_index("row").sort("row")
    campaign = np.full((n, TOP_K), "", dtype=object)
    campaign[pairs["row"].to_numpy(), pairs["slot"].to_numpy()] = pairs["C17"].to_numpy()
    return CandidateMatrices(
        rows=ordered,
        pairs=pairs,
        test_rows=test.height,
        served=served,
        candidates=candidates,
        mask=_matrix(pairs, "pctr", n, -1.0) >= 0,
        pctr=_matrix(pairs, "pctr", n, 0.0),
        reward=_matrix(pairs, "reward", n, 0.0),
        logit=_matrix(pairs, "logit", n, 0.0),
        gated=_matrix(pairs, "gated", n, 1.0).astype(bool),
        evidence=_matrix(pairs, "evidence", n, 0.0),
        campaign=campaign,
        logged=ordered["logged_slot"].to_numpy(),
        y=ordered["click"].to_numpy().astype(np.float64),
        mu=ordered["logged_mu"].to_numpy(),
        blocks=ordered["ts"].dt.epoch("s").to_numpy() // 3600,
    )


def run(
    settings: Settings | None = None, data_dir: Path | None = None, report: Path = REPORT, split: str = "test"
) -> list[Estimate]:
    """Evaluate policies on `split`; write ope.json and decisions.jsonl (test only) and the report."""
    settings = settings or get_settings()
    c = load_candidates(settings, data_dir, split)
    n = len(c.y)
    q_logged = c.at_logged(c.reward)
    results: list[Estimate] = [_observed(c.y, c.blocks)]
    lifts: dict[str, dict[str, tuple[float, float, float]]] = {}
    for name, pi in policies(c.candidates, n, settings).items():
        args = (c.at_logged(pi), c.mu, c.y, q_logged, (pi * c.reward).sum(axis=1), c.blocks)
        results += estimate(name, *args)
        lifts[name] = lift(*args)
    if split == "test":
        art = settings.artifacts_dir
        art.joinpath("ope.json").write_text(json.dumps({"policies": [r.model_dump() for r in results]}, indent=2))
        _write_decisions(c.candidates, settings, art / "decisions.jsonl")
    _write_report(results, lifts, c.test_rows, n, c.served, report, split)
    return results


def _observed(y: np.ndarray, blocks: np.ndarray) -> Estimate:
    ones = np.ones_like(y)
    est = estimate("logging (observed)", ones, ones, y, y, y, blocks)[0]
    return est.model_copy(update={"estimator": "on-policy"})


def _in_pair_order(
    candidates: dict[int, list[tuple[int, Candidate, str]]], pairs: pl.DataFrame
) -> list[tuple[int, Candidate, str]]:
    lookup = {(row, slot): (slot, c, tier) for row, entries in candidates.items() for slot, c, tier in entries}
    return [lookup[(row, slot)] for row, slot in pairs.select("row", "slot").iter_rows()]


def _write_decisions(candidates: dict[int, list[tuple[int, Candidate, str]]], settings: Settings, path: Path) -> None:
    rows = sorted(candidates)
    sample = sorted(
        np.random.default_rng(settings.seed).choice(rows, size=min(DECISION_SAMPLE, len(rows)), replace=False)
    )
    with path.open("w") as out:
        for row in sample:
            entries = candidates[int(row)]
            decision: Decision = decide([c for _, c, _ in entries], entries[0][2], f"ope-{row}", settings.policy)
            record = {
                "request_id": f"ope-{row}",
                "candidates": [
                    {
                        "candidate_id": r.candidate_id,
                        "pctr": r.pctr,
                        "gated": r.gated,
                        "gate_reasons": [str(g) for g in r.gate_reasons],
                        "propensity": r.propensity,
                    }
                    for r in decision.ranked
                ],
                "chosen_id": decision.chosen_id,
                "propensity": decision.propensity,
                "explored": decision.explored,
            }
            out.write(json.dumps(record) + "\n")


def _write_report(
    results: list[Estimate],
    lifts: dict[str, dict[str, tuple[float, float, float]]],
    test_rows: int,
    rows: int,
    served: pl.DataFrame,
    path: Path,
    split: str = "test",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cells = served.select(*CELL).n_unique()
    mean_k = float(served.group_by(CELL).len()["len"].mean())  # pyright: ignore[reportArgumentType]
    lines = [
        f"# Off-policy evaluation on the {split} split (generated by `uv run poe ope`)",
        "",
        "DR uses an independent LightGBM reward model (trained before the last training day), not the evaluated "
        "policy's own pCTR.",
        "",
        f"Evaluated impressions: {rows:,} of {test_rows:,} {split} rows ({rows / test_rows:.1%}); "
        f"cells: {cells:,}; mean candidates per cell: {mean_k:.2f}",
        "",
        "| Policy | Estimator | CTR | 95 % CI | ESS | Max weight |",
        "|---|---|---|---|---|---|",
        *[
            f"| {r.name} | {r.estimator} | {r.value:.4f} | [{r.ci_low:.4f}, {r.ci_high:.4f}] "
            f"| {r.ess:,.0f} | {r.max_weight:.1f} |"
            for r in results
        ],
        "",
        "## Lift over the logging policy (paired hour-block bootstrap)",
        "",
        "| Policy | SNIPS lift [95 % CI] | DR lift [95 % CI] |",
        "|---|---|---|",
        *[
            f"| {name} | {s[0]:+.4f} [{s[1]:+.4f}, {s[2]:+.4f}] | {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}] |"
            for name, v in lifts.items()
            for s, d in [(v["snips"], v["dr"])]
        ],
    ]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    import sys

    # `python -m charade.analysis.policy_eval [val|test]`: select on val, confirm on test.
    chosen = sys.argv[1] if len(sys.argv) > 1 else "test"
    target = REPORT if chosen == "test" else REPORT.with_name(f"ope_{chosen}.md")
    for estimate_ in run(report=target, split=chosen):
        print(estimate_.model_dump())
