"""Cold-start analysis (`uv run poe coldstart`) -> reports/coldstart/coldstart.md.

1. Which features carry signal before any click: permutation importance on cold slices
   (test rows whose character, or whose user, has no training history), measured as the NE
   increase when one field is shuffled across rows.
2. Bootstrapping prior: beta-binomial prior per genre x tier, fitted by the method of moments on
   characters' training CTRs. Its strength alpha + beta is the evidence a character needs before
   its own clicks outweigh the prior.
3. Graduation: does a character-level correction learned from its own earlier impressions improve
   the model, and from how many impressions on? On validation + test, in time order, each
   character's logit offset is a shrunk one-step Newton estimate from strictly earlier hours:
       delta = sum(y - p) / (sum p(1 - p) + tau)
   with tau = the prior strength. The NE of corrected vs uncorrected predictions, bucketed by the
   number of earlier impressions, shows where (if anywhere) per-character parameters pay off.
"""

from pathlib import Path

import numpy as np
import polars as pl

from charade.config import Settings, get_settings
from charade.evaluation.metrics import logloss_rows, normalized_entropy
from charade.features.spec import Encoded, encode
from charade.models.dataset import build_frame
from charade.scoring.scorer import Scorer

OUT = Path("reports/coldstart")
BUCKETS = (0, 5, 20, 50, 200, 1000)


def _table(frame: pl.DataFrame) -> str:
    head = "| " + " | ".join(frame.columns) + " |\n|" + "---|" * frame.width
    rows = [
        "| "
        + " | ".join(f"{v:.4f}" if isinstance(v, float) else f"{v:,}" if isinstance(v, int) else str(v) for v in r)
        + " |"
        for r in frame.iter_rows()
    ]
    return "\n".join([head, *rows])


def permutation_importance(scorer: Scorer, rows: pl.DataFrame, seed: int) -> pl.DataFrame:
    """NE increase per field when that field is shuffled within `rows`."""
    data = encode(scorer.spec, rows)
    y = rows["click"].to_numpy().astype(np.float64)
    base = normalized_entropy(y, scorer.pctr(data))
    rng = np.random.default_rng(seed)
    out: list[dict[str, object]] = []
    for i, field in enumerate(scorer.spec.categorical):
        cat = data.categorical.copy()
        cat[:, i] = rng.permutation(cat[:, i])
        out.append(
            {
                "feature": field.name,
                "delta_ne": normalized_entropy(y, scorer.pctr(Encoded(categorical=cat, dense=data.dense))) - base,
            }
        )
    for j, name in enumerate(scorer.spec.dense):
        dense = data.dense.copy()
        dense[:, j] = rng.permutation(dense[:, j])
        out.append(
            {
                "feature": name,
                "delta_ne": normalized_entropy(y, scorer.pctr(Encoded(categorical=data.categorical, dense=dense)))
                - base,
            }
        )
    return pl.DataFrame(out).sort("delta_ne", descending=True)


def beta_prior(train: pl.DataFrame, min_n: int = 50) -> pl.DataFrame:
    """Method-of-moments Beta(alpha, beta) per genre x tier over characters' training CTRs."""
    per_char = train.group_by("character_id", "genre", "safety_tier").agg(
        pl.len().alias("n"), pl.col("click").mean().alias("ctr")
    )
    per_char = per_char.filter(pl.col("n") >= min_n)
    rows: list[dict[str, object]] = []
    if per_char.height == 0:
        return pl.DataFrame(
            schema={
                "genre": pl.Utf8,
                "safety_tier": pl.Utf8,
                "characters": pl.Int64,
                "mean": pl.Float64,
                "true_sd": pl.Float64,
                "prior_strength": pl.Float64,
            }
        )
    for (genre, tier), part in per_char.group_by("genre", "safety_tier"):
        m = float(part["ctr"].mean())  # pyright: ignore[reportArgumentType]
        # Observed variance minus the binomial noise each character's CTR carries.
        noise = float((part["ctr"] * (1 - part["ctr"]) / part["n"]).mean())  # pyright: ignore[reportArgumentType]
        var = max(float(part["ctr"].var()) - noise, 1e-6)  # pyright: ignore[reportArgumentType]
        strength = max(m * (1 - m) / var - 1, 1.0)
        rows.append(
            {
                "genre": genre,
                "safety_tier": tier,
                "characters": part.height,
                "mean": m,
                "true_sd": var**0.5,
                "prior_strength": strength,
            }
        )
    return pl.DataFrame(rows).sort("genre", "safety_tier")


def pooled_strength(train: pl.DataFrame, min_n: int = 50) -> tuple[float, float]:
    """(true between-character sd beyond genre x tier, implied prior strength), pooled over all cells."""
    per_char = train.group_by("character_id", "genre", "safety_tier").agg(
        pl.len().alias("n"), pl.col("click").mean().alias("ctr")
    )
    per_char = per_char.filter(pl.col("n") >= min_n).with_columns(
        (pl.col("ctr") - pl.col("ctr").mean().over("genre", "safety_tier")).alias("resid"),
        (pl.col("ctr") * (1 - pl.col("ctr")) / pl.col("n")).alias("noise"),
    )
    if per_char.height < 2:
        return float("nan"), float("inf")
    var = max(float(per_char["resid"].var()) - float(per_char["noise"].mean()), 1e-8)  # pyright: ignore[reportArgumentType]
    m = float(per_char["ctr"].mean())  # pyright: ignore[reportArgumentType]
    return var**0.5, m * (1 - m) / var - 1


def graduation(frame: pl.DataFrame, p: np.ndarray, tau: float) -> pl.DataFrame:
    """NE with and without a causal per-character logit correction, by earlier-impression bucket."""
    rows = frame.with_columns(pl.Series("p", p)).sort("ts", "id")
    hourly = (
        rows.group_by("character_id", "ts")
        .agg(
            (pl.col("click") - pl.col("p")).sum().alias("r"),
            (pl.col("p") * (1 - pl.col("p"))).sum().alias("h"),
            pl.len().alias("k"),
        )
        .sort("character_id", "ts")
        .with_columns(
            (pl.col("r").cum_sum().over("character_id") - pl.col("r")).alias("R"),
            (pl.col("h").cum_sum().over("character_id") - pl.col("h")).alias("H"),
            (pl.col("k").cum_sum().over("character_id") - pl.col("k")).alias("earlier"),
        )
    )
    rows = rows.join(hourly.select("character_id", "ts", "R", "H", "earlier"), on=["character_id", "ts"])
    logit = np.log(rows["p"].to_numpy() / (1 - rows["p"].to_numpy()))
    delta = rows["R"].to_numpy() / (rows["H"].to_numpy() + tau)
    corrected = 1 / (1 + np.exp(-(logit + delta)))
    rows = rows.with_columns(pl.Series("q", corrected))
    labels = [f"<={b}" for b in BUCKETS[1:]] + [f">{BUCKETS[-1]}"]
    rows = rows.with_columns(
        pl.col("earlier").cut(list(BUCKETS[1:]), labels=labels, left_closed=False).cast(pl.Utf8).alias("bucket")
    )
    out: list[dict[str, object]] = []
    for label in labels:
        part = rows.filter(pl.col("bucket") == label)
        if part.height < 500:
            continue
        y = part["click"].to_numpy().astype(np.float64)
        base, corr = logloss_rows(y, part["p"].to_numpy()).mean(), logloss_rows(y, part["q"].to_numpy()).mean()
        out.append(
            {
                "earlier_impressions": label,
                "rows": part.height,
                "model_ne": normalized_entropy(y, part["p"].to_numpy()),
                "corrected_ne": normalized_entropy(y, part["q"].to_numpy()),
                "delta_logloss": float(corr - base),
            }
        )
    return pl.DataFrame(out)


def run(settings: Settings | None = None, data_dir: Path | None = None, out: Path = OUT) -> dict[str, pl.DataFrame]:
    """Write reports/coldstart/coldstart.md and return the tables."""
    settings = settings or get_settings()
    scorer = Scorer(settings.artifacts_dir)
    frame = build_frame(data_dir or settings.data_dir, None)
    train = frame.filter(pl.col("split") == "train")
    known = train["character_id"].unique().implode()
    test = frame.filter(pl.col("split") == "test")
    cold_char = test.filter(~pl.col("character_id").is_in(known))
    new_user = test.filter(pl.col("user_imps") == 0)
    prior = beta_prior(train)
    true_sd, tau = pooled_strength(train)
    later = frame.filter(pl.col("split") != "train")
    p_later = scorer.pctr(encode(scorer.spec, later))
    tables = {
        "Permutation importance, characters unseen in training (test)": permutation_importance(
            scorer, cold_char, settings.seed
        ).head(12),
        "Permutation importance, users with no earlier impressions (test)": permutation_importance(
            scorer, new_user, settings.seed
        ).head(12),
        "Beta prior per genre x tier (training characters with >= 50 impressions)": prior,
        f"Graduation, tau = {tau:,.0f} (pooled prior; true character sd beyond genre x tier = {true_sd:.4f})": (
            graduation(later, p_later, tau)
        ),
        "Graduation sensitivity, tau = 100 (as if characters varied by ~4 pp)": graduation(later, p_later, 100.0),
    }
    out.mkdir(parents=True, exist_ok=True)
    body = [
        "# Cold start (generated by `uv run poe coldstart`)",
        "",
        f"Cold-character test rows: {cold_char.height:,}; new-user test rows: {new_user.height:,}.",
        "",
    ]
    for title, table in tables.items():
        body += [f"## {title}", "", _table(table), ""]
    (out / "coldstart.md").write_text("\n".join(body))
    return tables


if __name__ == "__main__":
    run()
