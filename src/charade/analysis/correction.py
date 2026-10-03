"""Replay of the live (campaign, genre) correction (`uv run poe correction`) -> reports/policy/correction.md.

Question (E012, H12): does correcting the model's pCTR with a per-(campaign, genre) posterior learned from
earlier outcomes hold CTR, improve prediction of the ad actually shown, and give cold campaigns a
graduation rule?

Replay on the reconstructed candidate sets of `charade.analysis.policy_eval`, in hour order. Rows in hour
h are scored with pair sums built from logged rows in strictly earlier hours (the logs carry no order
within an hour). The observation stream is the logged ad's outcome and the model's pCTR for it, so the
state depends on logs, never on the simulated policy's choices, and the per-row target probabilities stay
valid for the doubly robust estimate. Each split starts from empty sums, which understates production,
where the state carries over.

Two readings per configuration:
    DR CTR lift vs greedy pCTR     same estimator and assumptions as the exposure-penalty study
    log loss on the logged ad      corrected vs raw pCTR for the ad that was shown: causal, online, and
                                   independent of the off-policy assumptions
Selection on the validation day by the pre-registered rule of `charade.analysis.adaptation`: among
deterministic configurations whose DR lower bound clears -0.2 pp, the largest point estimate. Test days
are read once to confirm.
"""

import itertools
import json
from pathlib import Path

import numpy as np
import polars as pl

from charade.analysis.policy_eval import CandidateMatrices, load_candidates
from charade.config import Settings, get_settings
from charade.evaluation.metrics import logloss_rows, normalized_entropy, paired_bootstrap
from charade.ranking.correction import graduated, posterior
from charade.scoring.scorer import Scorer, bundle_sha256

REPORT = Path("reports/policy/correction.md")
GRADUATION = Path("reports/coldstart/graduation.csv")
PRIORS = (5.0, 20.0, 80.0, 320.0)
HALF_LIVES: tuple[float | None, ...] = (12.0, 48.0, None)
ZS = (0.0, 1.0)
MAX_LOSS_PP = 0.2
"""Pre-registered non-inferiority margin: the 95 % lower bound of the DR CTR change must clear -0.2 pp."""


def replay(c: CandidateMatrices, prior: float, half_life: float | None, z: float) -> dict[str, np.ndarray]:
    """Run the corrected greedy policy over `c` in hour order.

    Returns per-row arrays: `chosen` slot (-1 when nothing is eligible), `p_adj` the corrected pCTR of the
    logged ad, `graduated` whether the logged ad's pair had graduated when the row was scored.
    """
    hours = c.rows["ts"].to_numpy().astype("datetime64[h]").astype(np.int64)
    genres = c.rows["genre"].to_list()
    n, k = c.pctr.shape
    eligible = c.mask & ~c.gated
    decay = 0.5 ** (1.0 / half_life) if half_life else 1.0
    keys = np.array([f"{c.campaign[t, s]}|{genres[t]}" for t in range(n) for s in range(k)], dtype=object)
    _, key_id = np.unique(keys, return_inverse=True)
    key_id = key_id.reshape(n, k)
    expected = np.zeros(int(key_id.max()) + 1)
    clicks = np.zeros_like(expected)
    chosen = np.full(n, -1, dtype=np.int64)
    p_adj = np.zeros(n)
    grad = np.zeros(n, dtype=bool)
    inside = c.logged >= 0
    logged = np.where(inside, c.logged, 0)
    for h in np.unique(hours):
        idx = np.flatnonzero(hours == h)
        mean, sd = posterior(expected[key_id[idx]], clicks[key_id[idx]], prior)
        score = np.clip(c.pctr[idx] * (mean + z * sd), 0.0, 1.0)
        score = np.where(eligible[idx], score + 1e-9 * np.tanh(c.logit[idx]), -np.inf)
        has = eligible[idx].any(axis=1)
        chosen[idx[has]] = np.argmax(score[has], axis=1)
        r = np.arange(len(idx))
        p_adj[idx] = np.clip(c.pctr[idx, logged[idx]] * mean[r, logged[idx]], 0.0, 1.0)
        grad[idx] = graduated(expected[key_id[idx]], prior)[r, logged[idx]]
        obs = idx[inside[idx]]
        np.add.at(expected, key_id[obs, c.logged[obs]], c.pctr[obs, c.logged[obs]])
        np.add.at(clicks, key_id[obs, c.logged[obs]], c.y[obs])
        expected *= decay
        clicks *= decay
    return {"chosen": chosen, "p_adj": p_adj, "graduated": grad}


def dr_rows(c: CandidateMatrices, chosen: np.ndarray) -> np.ndarray:
    """Per-row doubly robust value of a deterministic policy, with the independent reward model."""
    pi = np.zeros_like(c.reward)
    ok = chosen >= 0
    pi[np.arange(len(c.y))[ok], chosen[ok]] = 1.0
    return (pi * c.reward).sum(axis=1) + c.at_logged(pi) / c.mu * (c.y - c.at_logged(c.reward))


def greedy_slots(c: CandidateMatrices) -> np.ndarray:
    """Slot the uncorrected greedy policy serves per row (-1 when nothing is eligible)."""
    eligible = c.mask & ~c.gated
    key = np.where(eligible, c.pctr + 1e-9 * np.tanh(c.logit), -np.inf)
    return np.where(eligible.any(axis=1), np.argmax(key, axis=1), -1)


def concentration(c: CandidateMatrices, chosen: np.ndarray) -> dict[str, float]:
    """Cohort dominance, cohort HHI and repeat exposure of the chosen ads (as `charade.analysis.adaptation`)."""
    genres, users = c.rows["genre"].to_list(), c.rows["user"].to_list()
    served: dict[str, dict[str, int]] = {}
    seen: set[tuple[str, str]] = set()
    repeats = 0
    for t in np.flatnonzero(chosen >= 0):
        camp = str(c.campaign[t, chosen[t]])
        per = served.setdefault(genres[t], {})
        per[camp] = per.get(camp, 0) + 1
        repeats += (users[t], camp) in seen
        seen.add((users[t], camp))
    n = int((chosen >= 0).sum())
    hhi = sum(sum((x / sum(v.values())) ** 2 for x in v.values()) * sum(v.values()) for v in served.values())
    return {
        "cohort_dominance": sum(max(v.values()) for v in served.values()) / n,
        "cohort_hhi": hhi / n,
        "repeat_exposure": repeats / n,
    }


def cold_campaign(c: CandidateMatrices, known: set[str]) -> np.ndarray:
    """Per row: the logged ad's campaign never appeared in training."""
    slot = np.where(c.logged >= 0, c.logged, 0)
    return np.array([str(c.campaign[t, slot[t]]) not in known for t in range(len(c.y))])


def table(c: CandidateMatrices, known: set[str], priors: tuple[float, ...] = PRIORS) -> pl.DataFrame:
    """One row per configuration: CTR lift vs greedy, logged-ad log loss, cold-campaign calibration, graduation."""
    base = dr_rows(c, greedy_slots(c))
    inside = c.logged >= 0
    y, blocks = c.y[inside], c.blocks[inside]
    p_model = c.at_logged(c.pctr)[inside]
    cold = cold_campaign(c, known)[inside]
    rows: list[dict[str, object]] = []
    for prior, half_life, z in itertools.product(priors, HALF_LIVES, ZS):
        r = replay(c, prior, half_life, z)
        delta, low, high = paired_bootstrap(dr_rows(c, r["chosen"]) - base, c.blocks)
        p_adj = r["p_adj"][inside]
        ll, ll_low, ll_high = paired_bootstrap(logloss_rows(y, p_adj) - logloss_rows(y, p_model), blocks)
        rows.append(
            {
                "prior": prior,
                "half_life_h": half_life if half_life is not None else float("inf"),
                "z": z,
                "dr_delta_vs_greedy_pp": 100 * delta,
                "ci_low_pp": 100 * low,
                "ci_high_pp": 100 * high,
                "logloss_adj_minus_model": ll,
                "ll_ci_low": ll_low,
                "ll_ci_high": ll_high,
                "ne_model": normalized_entropy(y, p_model),
                "ne_adj": normalized_entropy(y, p_adj),
                "pred_obs_model_cold": float(p_model[cold].mean() / y[cold].mean()) if cold.any() else float("nan"),
                "pred_obs_adj_cold": float(p_adj[cold].mean() / y[cold].mean()) if cold.any() else float("nan"),
                "graduated_share_last_hour": float(r["graduated"][inside][blocks == blocks.max()].mean()),
                **concentration(c, r["chosen"]),
            }
        )
    return pl.DataFrame(rows)


def select(val: pl.DataFrame) -> tuple[float, float | None]:
    """Pre-registered: deterministic (z = 0) rows whose DR lower bound clears the margin; best point estimate.

    Returns (prior, half_life) or (nan, None) when nothing qualifies.
    """
    ok = val.filter((pl.col("z") == 0.0) & (pl.col("ci_low_pp") >= -MAX_LOSS_PP)).sort(
        "dr_delta_vs_greedy_pp", descending=True
    )
    if ok.height == 0:
        return float("nan"), None
    row = ok.row(0, named=True)
    half_life = float(row["half_life_h"])
    return float(row["prior"]), None if np.isinf(half_life) else half_life


def graduation_curve(c: CandidateMatrices, prior: float, half_life: float | None) -> pl.DataFrame:
    """Share of evaluated impressions whose logged pair had graduated, per hour of the split."""
    r = replay(c, prior, half_life, 0.0)
    inside = c.logged >= 0
    hours = c.rows["ts"].to_numpy().astype("datetime64[h]")[inside].astype(str)
    frame = pl.DataFrame({"hour": hours, "graduated": r["graduated"][inside]})
    return frame.group_by("hour").agg(pl.col("graduated").mean(), pl.len().alias("rows")).sort("hour")


def _markdown(frame: pl.DataFrame) -> str:
    head = "| " + " | ".join(frame.columns) + " |\n|" + "---|" * frame.width

    def cell(v: object) -> str:
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    body = "\n".join("| " + " | ".join(cell(v) for v in r) + " |" for r in frame.iter_rows())
    return f"{head}\n{body}"


def run(
    settings: Settings | None = None,
    data_dir: Path | None = None,
    report: Path = REPORT,
    graduation: Path = GRADUATION,
    priors: tuple[float, ...] = PRIORS,
) -> dict[str, pl.DataFrame]:
    """Select on validation, confirm on test, write the report and the graduation curve."""
    settings = settings or get_settings()
    known = set(Scorer(settings.artifacts_dir).spec.vocabularies["C17"])
    val_c = load_candidates(settings, data_dir, "val")
    val = table(val_c, known, priors)
    prior, half_life = select(val)
    test_c = load_candidates(settings, data_dir, "test")
    test = table(test_c, known, priors)
    report.parent.mkdir(parents=True, exist_ok=True)
    graduation.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Live (campaign, genre) correction (generated by `uv run poe correction`)",
        "",
        "Gamma-Poisson posterior on a pCTR multiplier per (campaign, genre), learned from logged outcomes in "
        "strictly earlier hours; greedy on the corrected score (z = 0) or on mean + z sd. DR with the independent "
        "reward model; log loss compares corrected vs raw pCTR on the ad actually shown. Each split starts from "
        "empty sums.",
        "",
        f"Selection on validation (pre-registered, 95 % lower bound of the DR change vs greedy >= -{MAX_LOSS_PP} pp, "
        f"best point estimate among z = 0): prior = {prior}, half-life = {half_life if half_life else 'none'}.",
        "",
        f"## Validation ({len(val_c.y):,} evaluated impressions)",
        "",
        _markdown(val),
        "",
        f"## Test ({len(test_c.y):,} evaluated impressions, read once)",
        "",
        _markdown(test),
        "",
    ]
    report.write_text("\n".join(lines))
    selected = test.filter((pl.col("prior") == prior) & (pl.col("z") == 0.0))
    if half_life is None:
        selected = selected.filter(pl.col("half_life_h").is_infinite())
    else:
        selected = selected.filter(pl.col("half_life_h") == half_life)
    summary = {
        "prior": prior,
        "half_life_hours": half_life,
        "test": selected.row(0, named=True) if selected.height else None,
        "bundle_sha256": bundle_sha256(settings.artifacts_dir),
    }
    report.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    curve = graduation_curve(test_c, prior, half_life) if np.isfinite(prior) else pl.DataFrame()
    curve.write_csv(graduation)
    return {"val": val, "test": test, "graduation": curve}


if __name__ == "__main__":
    for name, frame in run().items():
        print(name)
        print(frame)
