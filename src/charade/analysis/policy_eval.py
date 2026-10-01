"""Offline evaluation of ranking policies on the test days (`uv run poe ope`).

The logs show one ad per impression, so a candidate set has to be reconstructed. For every
publisher x hour cell (`site_id`, `app_id`, `ts`), the candidates are the creatives served in that
cell (`C14` + `banner_pos`, with the creative's other ad fields), capped at the `TOP_K` most
frequent. The logging propensity of an ad is its share of the cell's impressions. Each evaluated
impression keeps its own context and user, and every candidate is scored as if it had been shown
there (including its own user x campaign exposure count). Rows whose logged ad is outside the
cell's top-K, or in single-candidate cells, are excluded and the coverage is reported.

Policies: logging (observed CTR), uniform random, greedy pCTR, and the shipped policy (gates +
greedy with a 5 % Thompson bucket). Writes `ope.json` and `decisions.jsonl` (sampled decisions
produced by the serving code path, `charade.ranking.policy.decide`).
"""

import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl

from charade.config import Settings, get_settings
from charade.evaluation.ope import Estimate, estimate, lift
from charade.features.derive import derive
from charade.features.spec import encode
from charade.models.dataset import build_frame
from charade.ranking.evidence import Evidence
from charade.ranking.policy import TIER_ORDER, Candidate, decide
from charade.scoring.scorer import EVIDENCE_FILE, Scorer

TOP_K = 10
AD_FIELDS = ("C14", "banner_pos", "C15", "C16", "C17", "C18", "C19", "C21")
CELL = ("site_id", "app_id", "ts")
DECISION_SAMPLE = 3000
TS_DRAWS = 256
REPORT = Path("reports/policy/ope.md")


def candidate_sets(test: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(candidates per cell with `mu` and `slot`, evaluated rows with the logged slot)."""
    served = (
        test.group_by(*CELL, "C14", "banner_pos")
        .agg(pl.len().alias("count"), *[pl.col(f).mode().sort().first() for f in AD_FIELDS[2:]])
        .with_columns((pl.col("count") / pl.col("count").sum().over(CELL)).alias("mu"))
        .sort([*CELL, "count", "C14", "banner_pos"], descending=[False, False, False, True, False, False])
        .with_columns(pl.int_range(pl.len()).over(CELL).alias("slot"))
        .filter(pl.col("slot") < TOP_K)
        .with_columns(pl.len().over(CELL).alias("k"))
        .filter(pl.col("k") >= 2)
    )
    rows = test.join(
        served.select(*CELL, "C14", "banner_pos", "slot", "mu"), on=[*CELL, "C14", "banner_pos"], how="inner"
    )
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


def policies(
    pctr: np.ndarray, logit: np.ndarray, mask: np.ndarray, evidence: np.ndarray, gated: np.ndarray, settings: Settings
) -> dict[str, np.ndarray]:
    """pi(slot | row) for every policy, shape [rows, TOP_K]."""
    cfg = settings.policy
    rows = np.arange(len(pctr))
    uniform = mask / mask.sum(axis=1, keepdims=True)
    greedy = np.zeros_like(pctr)
    # Rank key = pCTR, ties (isotonic plateaus) broken by the raw logit, as in `decide`.
    key = pctr + 1e-9 * np.tanh(logit)
    greedy[rows, np.where(mask, key, -1).argmax(axis=1)] = 1.0
    open_mask = mask & ~gated
    has_open = open_mask.any(axis=1)
    shipped_greedy = np.zeros_like(pctr)
    shipped_greedy[rows, np.where(open_mask, key, -1).argmax(axis=1)] = 1.0
    n = np.clip(evidence, cfg.min_evidence, cfg.max_evidence)
    p = np.clip(pctr, 1e-4, 1 - 1e-4)
    rng = np.random.default_rng(settings.seed)
    draws = rng.beta(p * n, (1 - p) * n, size=(TS_DRAWS, *pctr.shape))
    draws = np.where(open_mask, draws, -1.0).argmax(axis=2)
    ts = np.stack([(draws == k).mean(axis=0) for k in range(TOP_K)], axis=1)
    shipped = ((1 - cfg.exploration_rate) * shipped_greedy + cfg.exploration_rate * ts) * has_open[:, None]
    return {"uniform random": uniform, "greedy pCTR": greedy, "shipped (gates + 5% Thompson)": shipped}


@dataclass
class CandidateMatrices:
    """Reconstructed candidate sets on the test days, as [rows, TOP_K] matrices (unused slots masked)."""

    rows: pl.DataFrame
    pairs: pl.DataFrame
    test_rows: int
    served: pl.DataFrame
    mask: np.ndarray
    pctr: np.ndarray
    logit: np.ndarray
    gated: np.ndarray
    evidence: np.ndarray
    campaign: np.ndarray
    logged: np.ndarray
    y: np.ndarray
    mu: np.ndarray
    blocks: np.ndarray


def load_candidates(settings: Settings, data_dir: Path | None = None) -> CandidateMatrices:
    """Build candidate sets, score every (impression, candidate) pair with the shipped bundle."""
    art = settings.artifacts_dir
    scorer = Scorer(art)
    evidence = Evidence.model_validate_json((art / EVIDENCE_FILE).read_text())
    frame = build_frame(data_dir or settings.data_dir, None)
    history = _campaign_exposure(frame)
    test = frame.filter(pl.col("split") == "test")
    served, rows = candidate_sets(test)
    pairs = expand(rows, served, history)
    logits, pctrs = scorer.score(encode(scorer.spec, pairs))
    pairs = pairs.with_columns(pl.Series("pctr", pctrs), pl.Series("logit", logits))
    pairs = pairs.with_columns(
        pl.struct("C17", "genre")
        .map_elements(lambda r: evidence.lookup(r["C17"], r["genre"]), return_dtype=pl.Float64)
        .alias("evidence"),
        _gated_expr(settings).alias("gated"),
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
        mask=_matrix(pairs, "pctr", n, -1.0) >= 0,
        pctr=_matrix(pairs, "pctr", n, 0.0),
        logit=_matrix(pairs, "logit", n, 0.0),
        gated=_matrix(pairs, "gated", n, 1.0).astype(bool),
        evidence=_matrix(pairs, "evidence", n, 0.0),
        campaign=campaign,
        logged=ordered["logged_slot"].to_numpy(),
        y=ordered["click"].to_numpy().astype(np.float64),
        mu=ordered["logged_mu"].to_numpy(),
        blocks=ordered["ts"].dt.epoch("s").to_numpy() // 3600,
    )


def run(settings: Settings | None = None, data_dir: Path | None = None, report: Path = REPORT) -> list[Estimate]:
    """Evaluate policies; write ope.json, decisions.jsonl and reports/policy/ope.md."""
    settings = settings or get_settings()
    c = load_candidates(settings, data_dir)
    n = len(c.y)
    q_logged = c.pctr[np.arange(n), c.logged]
    results: list[Estimate] = [_observed(c.y, c.blocks)]
    lifts: dict[str, dict[str, tuple[float, float, float]]] = {}
    for name, pi in policies(c.pctr, c.logit, c.mask, c.evidence, c.gated, settings).items():
        args = (pi[np.arange(n), c.logged], c.mu, c.y, q_logged, (pi * c.pctr).sum(axis=1), c.blocks)
        results += estimate(name, *args)
        lifts[name] = lift(*args)
    art = settings.artifacts_dir
    art.joinpath("ope.json").write_text(json.dumps({"policies": [r.model_dump() for r in results]}, indent=2))
    _write_decisions(c.pairs, c.rows, settings, art / "decisions.jsonl")
    _write_report(results, lifts, c.test_rows, n, c.served, report)
    return results


def _observed(y: np.ndarray, blocks: np.ndarray) -> Estimate:
    ones = np.ones_like(y)
    est = estimate("logging (observed)", ones, ones, y, y, y, blocks)[0]
    return est.model_copy(update={"estimator": "on-policy"})


def _gated_expr(settings: Settings) -> pl.Expr:
    cfg = settings.policy
    tier = pl.col("safety_tier").replace_strict(TIER_ORDER, return_dtype=pl.Int64)
    limit = pl.col("C21").replace_strict(
        {k: TIER_ORDER[v] for k, v in cfg.advertiser_max_tier.items()}, default=2, return_dtype=pl.Int64
    )
    return (tier > limit) | (pl.col("user_campaign_imps") >= cfg.frequency_cap)


def _write_decisions(pairs: pl.DataFrame, rows: pl.DataFrame, settings: Settings, path: Path) -> None:
    sample = rows.sample(min(DECISION_SAMPLE, rows.height), seed=settings.seed)["row"].to_list()
    by_row = pairs.filter(pl.col("row").is_in(sample)).partition_by("row", as_dict=True)
    with path.open("w") as out:
        for (row,), cands in sorted(by_row.items()):
            tier = cands["safety_tier"][0]
            candidates = [
                Candidate(
                    candidate_id=f"{r['C14']}@{r['banner_pos']}",
                    advertiser_id=r["C21"],
                    campaign_id=r["C17"],
                    pctr=r["pctr"],
                    evidence=r["evidence"],
                    prior_exposures=int(r["user_campaign_imps"]),
                    logit=r["logit"],
                )
                for r in cands.iter_rows(named=True)
            ]
            decision = decide(candidates, tier, f"ope-{row}", settings.policy)
            record = {
                "request_id": f"ope-{row}",
                "candidates": [
                    {
                        "candidate_id": r.candidate_id,
                        "pctr": r.pctr,
                        "gated": r.gated,
                        "gate_reasons": [str(g) for g in r.gate_reasons],
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
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cells = served.select(*CELL).n_unique()
    mean_k = float(served.group_by(CELL).len()["len"].mean())  # pyright: ignore[reportArgumentType]
    lines = [
        "# Off-policy evaluation (generated by `uv run poe ope`)",
        "",
        f"Evaluated impressions: {rows:,} of {test_rows:,} test rows ({rows / test_rows:.1%}); "
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
    for estimate_ in run():
        print(estimate_.model_dump())
