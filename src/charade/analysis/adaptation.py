"""Adaptation layer prototype (`uv run poe adapt`) -> reports/drift/adaptation.md.

Problem: greedy pCTR ranking concentrates a cohort (genre) on the few campaigns the model likes
best, so dominant cohorts see the same ads again and again. The layer re-balances with a
cohort-level exposure penalty that adapts as allocations shift:

    score(a) = pCTR(a) * exp(-lambda * share_{genre, campaign(a)})

where `share` is the campaign's share of the cohort's recent allocations (exponentially decayed
per cohort impression, half-life `HALF_LIFE`). The state depends only on the policy's own past
choices and on contexts, never on outcomes, so per-row target probabilities stay valid for
off-policy estimation. Replayed over the test days on the reconstructed candidate sets
(`charade.analysis.policy_eval`):

    DR CTR and its paired difference to greedy (hour-block bootstrap)
    cohort dominance: impression-weighted mean over genres of the top campaign's allocation share
    cohort HHI: concentration of each genre's allocation over campaigns
    repeat exposure: share of served impressions where this user already got the same campaign
"""

from pathlib import Path

import numpy as np
import polars as pl

from charade.analysis.policy_eval import CandidateMatrices, load_candidates
from charade.config import Settings, get_settings
from charade.evaluation.metrics import paired_bootstrap

OUT = Path("reports/drift/adaptation.md")
LAMBDAS = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0)
HALF_LIFE = 2000.0


def simulate(c: CandidateMatrices, lam: float, half_life: float = HALF_LIFE) -> tuple[np.ndarray, dict[str, float]]:
    """Run the policy over rows in time order; return the chosen slot per row and exposure metrics."""
    decay = 0.5 ** (1 / half_life)
    order = np.argsort(c.rows["ts"].to_numpy(), kind="stable")
    genres: list[str] = c.rows["genre"].to_list()
    users: list[str] = c.rows["user"].to_list()
    eligible = c.mask & ~c.gated
    key = np.where(eligible, c.pctr + 1e-9 * np.tanh(c.logit), -np.inf)
    chosen = np.full(len(order), -1, dtype=np.int64)
    raw: dict[str, dict[str, float]] = {}
    scale: dict[str, float] = {}
    total: dict[str, float] = {}
    served: dict[str, dict[str, int]] = {}
    seen: set[tuple[str, str]] = set()
    repeats = 0
    for t in order.tolist():
        if not eligible[t].any():
            continue
        g = genres[t]
        counts, s = raw.setdefault(g, {}), scale.setdefault(g, 1.0)
        tot = total.get(g, 0.0)
        if lam and tot > 0:
            shares = np.array([counts.get(str(camp), 0.0) * s / tot for camp in c.campaign[t]])
            slot = int(np.argmax(np.where(eligible[t], c.pctr[t] * np.exp(-lam * shares), -np.inf)))
        else:
            slot = int(np.argmax(key[t]))
        chosen[t] = slot
        camp = str(c.campaign[t, slot])
        s *= decay
        if s < 1e-12:  # renormalise the lazy decay
            raw[g] = {k: v * s for k, v in counts.items()}
            counts, s = raw[g], 1.0
        counts[camp] = counts.get(camp, 0.0) + 1 / s
        scale[g], total[g] = s, tot * decay + 1
        per = served.setdefault(g, {})
        per[camp] = per.get(camp, 0) + 1
        repeats += (users[t], camp) in seen
        seen.add((users[t], camp))
    n_served = int((chosen >= 0).sum())
    dominance = sum(max(v.values()) for v in served.values()) / n_served
    hhi = sum(sum((x / sum(v.values())) ** 2 for x in v.values()) * sum(v.values()) for v in served.values()) / n_served
    return chosen, {"cohort_dominance": dominance, "cohort_hhi": hhi, "repeat_exposure": repeats / n_served}


def _dr_rows(c: CandidateMatrices, chosen: np.ndarray) -> np.ndarray:
    n = len(c.y)
    pi = np.zeros_like(c.pctr)
    ok = chosen >= 0
    pi[np.arange(n)[ok], chosen[ok]] = 1.0
    w = pi[np.arange(n), c.logged] / c.mu
    q_logged = c.pctr[np.arange(n), c.logged]
    return (pi * c.pctr).sum(axis=1) + w * (c.y - q_logged)


def run(
    settings: Settings | None = None,
    data_dir: Path | None = None,
    out: Path = OUT,
    lambdas: tuple[float, ...] = LAMBDAS,
) -> pl.DataFrame:
    """Pareto table of DR CTR vs exposure concentration for each penalty strength."""
    settings = settings or get_settings()
    c = load_candidates(settings, data_dir)
    base_dr: np.ndarray | None = None
    rows: list[dict[str, object]] = []
    for lam in lambdas:
        chosen, exposure = simulate(c, lam)
        dr = _dr_rows(c, chosen)
        if base_dr is None:
            base_dr = dr
        delta, low, high = paired_bootstrap(dr - base_dr, c.blocks)
        rows.append(
            {
                "lambda": lam,
                "dr_ctr": float(dr.mean()),
                "dr_delta_vs_greedy": delta,
                "ci_low": low,
                "ci_high": high,
                **exposure,
            }
        )
    table = pl.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = "| " + " | ".join(table.columns) + " |\n|" + "---|" * table.width
    lines = ["| " + " | ".join(f"{v:.4f}" for v in r) + " |" for r in table.iter_rows()]
    out.write_text(
        "# Adaptation layer (generated by `uv run poe adapt`)\n\n"
        f"Greedy pCTR (lambda = 0) vs cohort-exposure penalty; {len(c.y):,} replayed test impressions; "
        f"half-life {HALF_LIFE:.0f} cohort impressions.\n\n" + header + "\n" + "\n".join(lines) + "\n"
    )
    return table


if __name__ == "__main__":
    print(run())
