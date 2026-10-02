"""Cold-start analysis (`uv run poe coldstart`) -> reports/coldstart/coldstart.md.

1. Which features carry signal before any click: permutation importance on cold slices
   (test rows whose character, or whose user, has no training history), measured as the NE
   increase when one field is shuffled across rows.
2. Spread beyond genre x tier: a beta-binomial prior per genre x tier, fitted by the method of moments
   on characters' training CTRs. Its strength alpha + beta says how much evidence a character would need
   before its own clicks outweigh its cell; a huge or infinite strength means no detectable spread.
3. Cold ads: test rows whose creative (C14) or campaign (C17) never appeared in training, which the
   standard slices (character, user) do not cover.
"""

from pathlib import Path

import numpy as np
import polars as pl

from charade.config import Settings, get_settings
from charade.evaluation.metrics import normalized_entropy
from charade.features.spec import Encoded, encode
from charade.models.dataset import build_frame
from charade.scoring.scorer import Scorer

OUT = Path("reports/coldstart")


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


def cold_ads(test: pl.DataFrame, train: pl.DataFrame, p: np.ndarray) -> pl.DataFrame:
    """NE and calibration on test rows by whether their creative and campaign appeared in training."""
    rows = test.with_columns(
        pl.Series("p", p),
        pl.col("C14").is_in(train["C14"].unique().implode()).alias("seen_creative"),
        pl.col("C17").is_in(train["C17"].unique().implode()).alias("seen_campaign"),
    )
    out: list[dict[str, object]] = []
    for name, part in (
        ("all test rows", rows),
        ("creative unseen in training", rows.filter(~pl.col("seen_creative"))),
        ("campaign unseen in training", rows.filter(~pl.col("seen_campaign"))),
        ("creative and campaign seen", rows.filter(pl.col("seen_creative") & pl.col("seen_campaign"))),
    ):
        y, q = part["click"].to_numpy().astype(np.float64), part["p"].to_numpy()
        out.append(
            {
                "slice": name,
                "rows": part.height,
                "share": part.height / rows.height,
                "ne": normalized_entropy(y, q),
                "pred_over_obs": float(q.mean() / y.mean()),
            }
        )
    return pl.DataFrame(out)


def run(settings: Settings | None = None, data_dir: Path | None = None, out: Path = OUT) -> dict[str, pl.DataFrame]:
    """Write reports/coldstart/coldstart.md and return the tables."""
    settings = settings or get_settings()
    scorer = Scorer(settings.artifacts_dir)
    frame = build_frame(settings, data_dir or settings.data_dir)
    train = frame.filter(pl.col("split") == "train")
    known = train["character_id"].unique().implode()
    test = frame.filter(pl.col("split") == "test")
    cold_char = test.filter(~pl.col("character_id").is_in(known))
    new_user = test.filter(pl.col("user_imps") == 0)
    prior = beta_prior(train)
    tables = {
        "Permutation importance, characters unseen in training (test)": permutation_importance(
            scorer, cold_char, settings.seed
        ).head(12),
        "Permutation importance, users with no earlier impressions (test)": permutation_importance(
            scorer, new_user, settings.seed
        ).head(12),
        "Beta prior per genre x tier (training characters with >= 50 impressions)": prior,
        "Cold ads (test): creatives and campaigns never seen in training": cold_ads(
            test, train, scorer.pctr(encode(scorer.spec, test))
        ),
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
