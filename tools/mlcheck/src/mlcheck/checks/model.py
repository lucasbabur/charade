"""Model quality checks, recomputed from raw predictions rather than trusted from reports."""

import numpy as np
import polars as pl

from mlcheck import stats
from mlcheck.context import Context
from mlcheck.registry import check
from mlcheck.result import Outcome, Severity, Stage, failed, passed

_REQUIRED_COLUMNS = ("id", "split", "model", "label", "pred", "ts")


def _rows(ctx: Context, model: str, split: str) -> pl.DataFrame:
    frame = ctx.predictions.filter((pl.col("model") == model) & (pl.col("split") == split))
    if frame.height == 0:
        raise ValueError(f"no predictions for model={model} split={split}")
    return frame


def _arrays(frame: pl.DataFrame) -> tuple[stats.Floats, stats.Floats]:
    return frame["label"].cast(pl.Float64).to_numpy(), frame["pred"].cast(pl.Float64).to_numpy()


def _holdout(ctx: Context) -> pl.DataFrame:
    return _rows(ctx, ctx.config.primary_model, ctx.config.holdout_split)


@check(
    "MLM001",
    "predictions-contract",
    Stage.MODEL,
    "Every gate below is computed from predictions.parquet; it must hold primary and baseline predictions on every "
    "evaluation split for the same ids.",
)
def predictions_contract(ctx: Context) -> Outcome:
    """Columns exist, labels are 0/1, and primary/baseline cover identical ids per evaluation split."""
    frame = ctx.predictions
    missing = [c for c in _REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        return failed("predictions.parquet missing columns", missing)
    if not set(frame["label"].unique().to_list()) <= {0, 1}:
        return failed("labels outside {0,1}")
    problems: list[str] = []
    for split in ctx.config.split_order[1:]:
        primary = set(_rows(ctx, ctx.config.primary_model, split)["id"].to_list())
        baseline = set(_rows(ctx, ctx.config.baseline_model, split)["id"].to_list())
        if primary != baseline:
            problems.append(f"{split}: {len(primary ^ baseline):,} ids not paired between primary and baseline")
    return failed("unpaired predictions", problems) if problems else passed(f"{frame.height:,} prediction rows")


@check(
    "MLM002",
    "prediction-sanity",
    Stage.MODEL,
    "Probabilities must be finite, strictly inside (0,1) and not constant; a 0 or 1 makes log loss infinite and a "
    "constant model ranks nothing.",
)
def prediction_sanity(ctx: Context) -> Outcome:
    """Every model's predictions are finite, in (0,1) and non-constant."""
    problems: list[str] = []
    for (model,), frame in ctx.predictions.group_by("model"):
        pred = frame["pred"].cast(pl.Float64).to_numpy()
        if not np.isfinite(pred).all():
            problems.append(f"{model}: non-finite predictions")
        elif (pred <= 0).any() or (pred >= 1).any():
            problems.append(f"{model}: predictions outside (0,1)")
        elif model != ctx.config.baseline_model and pred.std() < 1e-6:
            problems.append(f"{model}: constant predictions")
    return failed("invalid predictions", problems) if problems else passed("predictions finite, in (0,1), varied")


@check(
    "MLM003",
    "holdout-normalized-entropy",
    Stage.MODEL,
    "Normalized entropy (log loss / base-rate entropy) is the metric the auction consumes; it must clear the floor "
    "on the untouched holdout.",
)
def holdout_normalized_entropy(ctx: Context) -> Outcome:
    """Primary model NE on the holdout is at most `max_test_ne`."""
    y, p = _arrays(_holdout(ctx))
    ne = stats.normalized_entropy(y, p)
    message = f"holdout NE {ne:.4f}, log loss {stats.log_loss(y, p):.5f}, n={len(y):,}"
    return failed(f"{message} > {ctx.thresholds.max_test_ne}") if ne > ctx.thresholds.max_test_ne else passed(message)


@check(
    "MLM004",
    "beats-baseline",
    Stage.MODEL,
    "The primary model must beat the baseline on the holdout with a paired, hour-block bootstrap CI that excludes "
    "zero; a point estimate alone can be noise.",
)
def beats_baseline(ctx: Context) -> Outcome:
    """Upper CI bound of (primary - baseline) holdout log loss is below zero."""
    primary = _holdout(ctx).select("id", "label", "pred", "ts")
    baseline = _rows(ctx, ctx.config.baseline_model, ctx.config.holdout_split).select("id", pl.col("pred").alias("b"))
    paired = primary.join(baseline, on="id", how="inner")
    y, p = _arrays(paired)
    b = paired["b"].cast(pl.Float64).to_numpy()
    diff = stats.log_loss_per_row(y, p) - stats.log_loss_per_row(y, b)
    blocks = paired["ts"].dt.truncate("1h").rank("dense").cast(pl.Int64).to_numpy()
    mean, low, high = stats.paired_block_bootstrap(
        diff, blocks, ctx.thresholds.bootstrap_resamples, ctx.thresholds.confidence
    )
    message = f"delta log loss {mean:+.5f} [{low:+.5f}, {high:+.5f}] over {len(np.unique(blocks))} hour blocks"
    return passed(message) if high < 0 else failed(f"{message}; CI does not exclude zero")


@check(
    "MLM005",
    "calibration-ratio",
    Stage.MODEL,
    "Ranking by pCTR x bid needs calibrated probabilities; the mean prediction must match the observed holdout rate.",
)
def calibration_ratio(ctx: Context) -> Outcome:
    """Holdout predicted/observed ratio within [calibration_low, calibration_high]."""
    y, p = _arrays(_holdout(ctx))
    ratio = stats.calibration_ratio(y, p)
    low, high = ctx.thresholds.calibration_low, ctx.thresholds.calibration_high
    message = f"predicted/observed {ratio:.4f} (pred {p.mean():.4%}, obs {y.mean():.4%})"
    return passed(message) if low <= ratio <= high else failed(f"{message} outside [{low}, {high}]")


@check(
    "MLM006",
    "calibration-per-period",
    Stage.MODEL,
    "Aggregate calibration can hide daily swings; each evaluation day must stay calibrated or drift needs an online "
    "correction.",
    Severity.WARNING,
)
def calibration_per_period(ctx: Context) -> Outcome:
    """Predicted/observed ratio per evaluation day within band."""
    evaluation = ctx.predictions.filter(
        (pl.col("model") == ctx.config.primary_model) & pl.col("split").is_in(ctx.config.split_order[1:])
    )
    daily = (
        evaluation.group_by(pl.col("ts").dt.date().alias("day"))
        .agg(pl.col("pred").mean().alias("p"), pl.col("label").mean().alias("y"), pl.len().alias("n"))
        .sort("day")
        .with_columns((pl.col("p") / pl.col("y")).alias("ratio"))
    )
    rows = [f"{r['day']}: {r['ratio']:.3f} (n={r['n']:,})" for r in daily.iter_rows(named=True)]
    low, high = ctx.thresholds.calibration_low, ctx.thresholds.calibration_high
    bad = daily.filter(~pl.col("ratio").is_between(low, high))
    return (
        failed(f"{bad.height} days outside [{low}, {high}]", rows)
        if bad.height
        else passed("all days calibrated", rows)
    )


@check(
    "MLM007",
    "expected-calibration-error",
    Stage.MODEL,
    "The ratio can be 1.0 while low and high scores are both wrong; ECE checks calibration across the score range.",
)
def expected_calibration_error(ctx: Context) -> Outcome:
    """Holdout equal-mass ECE at most `max_ece`."""
    y, p = _arrays(_holdout(ctx))
    ece = stats.expected_calibration_error(y, p, ctx.thresholds.ece_bins)
    message = f"ECE {ece:.4f} over {ctx.thresholds.ece_bins} equal-mass bins"
    return passed(message) if ece <= ctx.thresholds.max_ece else failed(f"{message} > {ctx.thresholds.max_ece}")


@check(
    "MLM008",
    "required-slices",
    Stage.MODEL,
    "Averages hide cold-start and minority-segment failures; required slices must be present and populated.",
)
def required_slices(ctx: Context) -> Outcome:
    """Every required slice column exists on holdout predictions and labels every row."""
    frame = _holdout(ctx)
    problems: list[str] = []
    for name in ctx.config.required_slices:
        column = f"slice_{name}"
        if column not in frame.columns:
            problems.append(f"{column} missing")
        elif nulls := frame[column].null_count():
            problems.append(f"{column} unlabeled on {nulls:,} rows")
    if problems:
        return failed("slice coverage incomplete", problems)
    return passed(f"{len(ctx.config.required_slices)} slices present")


@check(
    "MLM009",
    "slice-normalized-entropy",
    Stage.MODEL,
    "A slice where the model is worse than predicting its own base rate is a segment the model actively hurts.",
    Severity.WARNING,
)
def slice_normalized_entropy(ctx: Context) -> Outcome:
    """Every slice value with at least `min_slice_rows` holdout rows has NE below 1."""
    frame = _holdout(ctx)
    report: list[str] = []
    bad = 0
    for name in [n for n in ctx.config.required_slices if f"slice_{n}" in frame.columns]:
        for (value,), part in frame.group_by(f"slice_{name}"):
            if part.height < ctx.thresholds.min_slice_rows:
                continue
            ne = stats.normalized_entropy(*_arrays(part))
            bad += ne >= 1
            report.append(
                f"{name}={value}: NE {ne:.4f} n={part.height:,}{'  <-- worse than base rate' if ne >= 1 else ''}"
            )
    report.sort()
    return failed(f"{bad} slices with NE >= 1", report) if bad else passed("every slice beats its base rate", report)
