"""Temporal drift (`uv run poe drift`) -> drift.json (mlcheck MLX001) and reports/drift/drift.md.

- PSI of every model input per day against the training window (categoricals: train top-50 values
  + other; dense: train deciles).
- CTR by day overall and by genre; character churn (first-seen share, top-100 Jaccard to the previous
  day, HHI of character share); ad rotation (share of impressions on creatives unseen before that day).
- Staleness: models trained up to day d (early-stopped on d + 1) scored on every later day. NE as a
  function of model age, against a model retrained daily.
- Online recalibration: a causal hourly logit offset per genre from exponentially weighted earlier
  residuals. Does it repair day-level shifts the frozen calibrator cannot?
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from charade.config import Settings, get_settings
from charade.evaluation.metrics import normalized_entropy
from charade.features.spec import Group, encode
from charade.models.core import train_dcn
from charade.models.dataset import build_frame
from charade.models.trainer import predict_logits
from charade.scoring.scorer import Scorer

OUT = Path("reports/drift")
TOP_VALUES = 50
EPS = 1e-4
HALF_LIFE_HOURS = 6.0
RECAL_TITLE = "Online recalibration (causal per-genre hourly offset, half-life {:.0f} h), validation + test"


def _table(frame: pl.DataFrame) -> str:
    head = "| " + " | ".join(frame.columns) + " |\n|" + "---|" * frame.width
    rows = [
        "| "
        + " | ".join(f"{v:.4f}" if isinstance(v, float) else f"{v:,}" if isinstance(v, int) else str(v) for v in r)
        + " |"
        for r in frame.iter_rows()
    ]
    return "\n".join([head, *rows])


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    """Population stability index between two count vectors over the same bins."""
    e = np.clip(expected / expected.sum(), EPS, None)
    a = np.clip(actual / actual.sum(), EPS, None)
    return float(((a - e) * np.log(a / e)).sum())


def psi_table(frame: pl.DataFrame, categorical: list[str], dense: list[str]) -> dict[str, dict[str, float]]:
    """PSI per feature per day (every day, including training days) against the training window."""
    train = frame.filter(pl.col("split") == "train")
    days = frame.with_columns(pl.col("ts").dt.date().cast(pl.Utf8).alias("day"))
    out: dict[str, dict[str, float]] = {}
    for name in categorical:
        top = (
            train[name].value_counts().sort(["count", name], descending=[True, False]).head(TOP_VALUES)[name].to_list()
        )
        binned = days.with_columns(
            pl.when(pl.col(name).is_in(top)).then(pl.col(name)).otherwise(pl.lit("__other__")).alias("bin")
        )
        ref = binned.filter(pl.col("split") == "train")["bin"].value_counts()
        out[name] = _psi_by_day(binned, ref)
    for name in dense:
        values = train[name].cast(pl.Float64).fill_null(0.0).to_numpy()
        edges = np.unique(np.quantile(values, np.linspace(0.1, 0.9, 9)))
        binned = days.with_columns(
            pl.col(name)
            .cast(pl.Float64)
            .fill_null(0.0)
            .map_batches(lambda s, e=edges: pl.Series(np.searchsorted(e, s.to_numpy(), side="right")))
            .alias("bin")
        )
        ref = binned.filter(pl.col("split") == "train")["bin"].value_counts()
        out[name] = _psi_by_day(binned, ref)
    return out


def _psi_by_day(binned: pl.DataFrame, ref: pl.DataFrame) -> dict[str, float]:
    result: dict[str, float] = {}
    for (day,), part in binned.group_by("day"):
        counts = part["bin"].value_counts().join(ref, on="bin", how="full", coalesce=True, suffix="_ref").fill_null(0)
        result[str(day)] = psi(counts["count_ref"].to_numpy().astype(float), counts["count"].to_numpy().astype(float))
    return dict(sorted(result.items()))


def churn(frame: pl.DataFrame) -> pl.DataFrame:
    """Per day: CTR, first-seen character share, top-100 Jaccard to the previous day, HHI, unseen-creative share."""
    days = frame.with_columns(pl.col("ts").dt.date().alias("day"))
    first_char = days.group_by("character_id").agg(pl.col("day").min().alias("first_char"))
    first_ad = days.group_by("C14").agg(pl.col("day").min().alias("first_ad"))
    days = days.join(first_char, on="character_id").join(first_ad, on="C14")
    rows: list[dict[str, object]] = []
    previous: set[str] = set()
    for (day,), part in sorted(days.group_by("day"), key=lambda kv: kv[0]):
        share = (
            part.group_by("character_id")
            .len()
            .with_columns((pl.col("len") / part.height).alias("s"))
            .sort(["len", "character_id"], descending=[True, False])  # id breaks ties: deterministic top-100
        )
        top = set(share.head(100)["character_id"].to_list())
        rows.append(
            {
                "day": str(day),
                "impressions": part.height,
                "ctr": float(part["click"].mean()),  # pyright: ignore[reportArgumentType]
                "first_seen_character_share": float((part["first_char"] == day).mean()),  # pyright: ignore[reportArgumentType]
                "top100_jaccard_prev_day": len(top & previous) / len(top | previous) if previous else float("nan"),
                "character_hhi": float((share["s"] ** 2).sum()),
                "new_creative_share": float((part["first_ad"] == day).mean()),  # pyright: ignore[reportArgumentType]
            }
        )
        previous = top
    return pl.DataFrame(rows)


STALE_TITLE = "Staleness: frozen models trained on equal 3-day windows, scored on later days (2 seeds averaged)"


def staleness(
    settings: Settings, frame: pl.DataFrame, window_days: int = 3, seeds: tuple[int, ...] = (0, 1)
) -> pl.DataFrame:
    """NE on each later day for models trained on the `window_days` days before day d, early-stopped on d.

    Every model sees the same amount of training data, so age is not confounded with volume: an
    earlier version trained on all days < d, which gave older models fewer days and showed a staleness
    slope that was really a data-volume effect. NE is averaged over `seeds`.
    """
    groups = {Group(g) for g in settings.model.groups}
    stamps: list[datetime] = frame["ts"].to_list()
    day0 = min(stamps).replace(hour=0)
    n_days = (max(stamps) - day0).days + 1
    rows: list[dict[str, object]] = []
    for cut in range(window_days, n_days - 1):
        val_day = day0 + timedelta(days=cut)
        fold = frame.filter(pl.col("ts") >= val_day - timedelta(days=window_days)).with_columns(
            pl.when(pl.col("ts") < val_day)
            .then(pl.lit("train"))
            .when(pl.col("ts") < val_day + timedelta(days=1))
            .then(pl.lit("val"))
            .otherwise(pl.lit("test"))
            .alias("split")
        )
        from charade.models.core import prepare  # noqa: PLC0415

        prep = prepare(fold, groups, text=False)
        later = fold.filter(pl.col("split") == "test").with_columns(pl.col("ts").dt.date().alias("day"))
        encoded = encode(prep.spec, later)
        logit = np.mean([predict_logits(train_dcn(prep, settings.model.dcn, s).model, encoded) for s in seeds], axis=0)
        later = later.with_columns(pl.Series("p", 1 / (1 + np.exp(-logit))))
        for (day,), part in later.group_by("day"):
            y = part["click"].to_numpy().astype(np.float64)
            rows.append(
                {
                    "trained_through": str((val_day - timedelta(days=1)).date()),
                    "scored_day": str(day),
                    "age_days": (day - val_day.date()).days + 1,
                    "ne": normalized_entropy(y, part["p"].to_numpy()),
                    "calibration_ratio": float(part["p"].to_numpy().mean() / y.mean()),
                }
            )
    return pl.DataFrame(rows).sort("scored_day", "age_days")


def age_slope(table: pl.DataFrame) -> float:
    """NE change per day of model age, within each scored day (so day-level difficulty cancels)."""
    centred = table.with_columns(
        (pl.col("ne") - pl.col("ne").mean().over("scored_day")).alias("dne"),
        (pl.col("age_days") - pl.col("age_days").mean().over("scored_day")).cast(pl.Float64).alias("dage"),
    )
    x, y = centred["dage"].to_numpy(), centred["dne"].to_numpy()
    return float((x * y).sum() / max((x * x).sum(), 1e-12))


def recalibration(frame: pl.DataFrame, p: np.ndarray, half_life: float = HALF_LIFE_HOURS) -> pl.DataFrame:
    """Daily NE and calibration of the frozen model vs a causal hourly per-genre logit offset."""
    decay = 0.5 ** (1 / half_life)
    rows = frame.with_columns(pl.Series("p", p))
    hourly = (
        rows.group_by("genre", "ts")
        .agg((pl.col("click") - pl.col("p")).sum().alias("r"), (pl.col("p") * (1 - pl.col("p"))).sum().alias("h"))
        .sort("genre", "ts")
    )
    offsets: list[pl.DataFrame] = []
    for _, part in hourly.group_by("genre"):
        r_acc, h_acc, out = 0.0, 0.0, []
        for r, h in zip(part["r"].to_list(), part["h"].to_list(), strict=True):
            out.append(r_acc / (h_acc + 50.0))  # strictly earlier hours, shrunk toward 0
            r_acc, h_acc = decay * r_acc + r, decay * h_acc + h
        offsets.append(part.select("genre", "ts").with_columns(pl.Series("offset", out)))
    rows = rows.join(pl.concat(offsets), on=["genre", "ts"])
    logit = np.log(rows["p"].to_numpy() / (1 - rows["p"].to_numpy()))
    rows = rows.with_columns(
        pl.Series("q", 1 / (1 + np.exp(-(logit + rows["offset"].to_numpy())))), pl.col("ts").dt.date().alias("day")
    )
    result: list[dict[str, object]] = []
    for (day,), part in sorted(rows.group_by("day"), key=lambda kv: kv[0]):
        y = part["click"].to_numpy().astype(np.float64)
        result.append(
            {
                "day": str(day),
                "frozen_ne": normalized_entropy(y, part["p"].to_numpy()),
                "recalibrated_ne": normalized_entropy(y, part["q"].to_numpy()),
                "frozen_ratio": float(part["p"].to_numpy().mean() / y.mean()),
                "recalibrated_ratio": float(part["q"].to_numpy().mean() / y.mean()),
            }
        )
    return pl.DataFrame(result)


def run(
    settings: Settings | None = None, data_dir: Path | None = None, out: Path = OUT, window_days: int = 3
) -> dict[str, pl.DataFrame]:
    """Write drift.json and reports/drift/drift.md."""
    settings = settings or get_settings()
    scorer = Scorer(settings.artifacts_dir)
    frame = build_frame(settings, data_dir or settings.data_dir)
    categorical = [f.name for f in scorer.spec.categorical]
    psi = psi_table(frame, categorical, scorer.spec.dense)
    after = sorted(
        {d for v in psi.values() for d in v}
        - set(frame.filter(pl.col("split") == "train")["ts"].dt.date().cast(pl.Utf8).unique().to_list())
    )
    (settings.artifacts_dir / "drift.json").write_text(
        json.dumps(
            {"reference": "train", "psi": {f: {d: v[d] for d in after if d in v} for f, v in psi.items()}}, indent=2
        )
    )
    worst = (
        pl.DataFrame([{"feature": f, "day": d, "psi": v} for f, days in psi.items() for d, v in days.items()])
        .filter(pl.col("day").is_in(after))
        .group_by("feature")
        .agg(pl.col("psi").max().alias("max_psi_after_training"))
        .sort("max_psi_after_training", descending=True)
    )
    later = frame.filter(pl.col("split") != "train")
    p = scorer.pctr(encode(scorer.spec, later))
    genre_ctr = (
        frame.group_by(pl.col("ts").dt.date().cast(pl.Utf8).alias("day"), "genre")
        .agg(pl.col("click").mean())
        .pivot(on="genre", index="day", values="click", sort_columns=True)
        .sort("day")
    )
    tables = {
        "Daily mix and churn": churn(frame),
        "CTR by day and genre": genre_ctr,
        "Most-drifted features (max daily PSI after the training window)": worst.head(12),
        STALE_TITLE: (stale := staleness(settings, frame, window_days)),
        RECAL_TITLE.format(HALF_LIFE_HOURS): recalibration(later, p),
    }
    out.mkdir(parents=True, exist_ok=True)
    body = ["# Drift (generated by `uv run poe drift`)", ""]
    for title, table in tables.items():
        body += [f"## {title}", "", _table(table), ""]
        if title == STALE_TITLE:
            body += [f"NE change per day of model age, within scored day: {age_slope(stale):+.5f}", ""]
    (out / "drift.md").write_text("\n".join(body))
    return tables


if __name__ == "__main__":
    run()
