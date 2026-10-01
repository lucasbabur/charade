"""Data contract checks on the raw event table and its entity references."""

import numpy as np
import polars as pl

from mlcheck.context import Context
from mlcheck.registry import check
from mlcheck.result import Outcome, Severity, Stage, failed, passed

_MAD_TO_SIGMA = 1.4826


def _expected_columns(ctx: Context) -> list[str]:
    cfg = ctx.data_config
    keys = [fk.column for fk in cfg.foreign_keys]
    return list(dict.fromkeys([cfg.id, cfg.label, cfg.time, *cfg.required_columns, *cfg.categorical, *keys]))


@check(
    "MLD001",
    "required-columns",
    Stage.DATA,
    "Every downstream step assumes these columns; a renamed upstream column must fail here, not as a silent OOV.",
)
def required_columns(ctx: Context) -> Outcome:
    """All declared columns exist."""
    missing = [c for c in _expected_columns(ctx) if c not in ctx.events.columns]
    if missing:
        return failed(f"{len(missing)} declared columns missing", missing)
    return passed(f"{len(_expected_columns(ctx))} declared columns present; {ctx.events.height:,} rows")


@check("MLD002", "binary-label", Stage.DATA, "The label must be exactly {0,1} and both classes must occur.")
def binary_label(ctx: Context) -> Outcome:
    """Label values are 0/1 and the positive rate is strictly between 0 and 1."""
    label = ctx.events[ctx.data_config.label]
    values = set(label.drop_nulls().unique().to_list())
    if not values <= {"0", "1"}:
        return failed(f"label has values outside {{0,1}}: {sorted(values)[:10]}")
    rate = (label == "1").mean()
    if values != {"0", "1"}:
        return failed(f"label has a single class: {values}")
    return passed(f"positive rate {rate:.4%}")


@check("MLD003", "unique-ids", Stage.DATA, "Duplicate event ids double-count clicks and can straddle a split.")
def unique_ids(ctx: Context) -> Outcome:
    """Event ids are unique."""
    ids = ctx.events[ctx.data_config.id]
    duplicates = ids.len() - ids.n_unique()
    return failed(f"{duplicates:,} duplicate ids") if duplicates else passed(f"{ids.len():,} unique ids")


@check("MLD004", "null-budget", Stage.DATA, "Nulls in declared columns must stay within the configured budget.")
def null_budget(ctx: Context) -> Outcome:
    """Null fraction per declared column is within `max_null_fraction`."""
    present = [c for c in _expected_columns(ctx) if c in ctx.events.columns]
    fractions = ctx.events.select(pl.col(present).null_count() / ctx.events.height).row(0, named=True)
    over = {c: f for c, f in fractions.items() if f > ctx.thresholds.max_null_fraction}
    if over:
        return failed(f"{len(over)} columns over null budget", [f"{c}: {f:.4%}" for c, f in over.items()])
    return passed(f"{len(present)} columns within null budget {ctx.thresholds.max_null_fraction:.2%}")


@check(
    "MLD005",
    "time-parses",
    Stage.DATA,
    "Every split, counter and drift window is keyed on event time; an unparseable timestamp silently drops rows.",
)
def time_parses(ctx: Context) -> Outcome:
    """Every timestamp parses with the declared format."""
    cfg = ctx.data_config
    bad = ctx.events.filter(pl.col(cfg.time).is_not_null() & pl.col("__ts").is_null())
    if bad.height:
        return failed(f"{bad.height:,} unparseable timestamps", bad[cfg.time].head(10).to_list())
    ts = ctx.events["__ts"]
    return passed(f"range {ts.min()} to {ts.max()}")


@check(
    "MLD006",
    "period-volume",
    Stage.DATA,
    "A period with abnormal volume (outage, partial day, duplicate load) distorts per-period metrics and drift.",
    Severity.WARNING,
)
def period_volume(ctx: Context) -> Outcome:
    """Rows per period stay within median +- k robust standard deviations."""
    counts = ctx.events.group_by(pl.col("__ts").dt.truncate(ctx.data_config.period)).len().sort("__ts")
    values = counts["len"].cast(pl.Float64).to_numpy()
    median = float(np.median(values))
    spread = _MAD_TO_SIGMA * float(np.median(np.abs(values - median)))
    limit = ctx.thresholds.volume_mad_k * spread
    outliers = counts.filter((pl.col("len") - median).abs() > limit)
    if outliers.height:
        rows = [f"{r['__ts']}: {r['len']:,} rows (median {median:,.0f})" for r in outliers.iter_rows(named=True)]
        return failed(f"{outliers.height} periods with abnormal volume", rows)
    return passed(f"{counts.height} periods, median {median:,.0f} rows")


@check(
    "MLD007",
    "dominant-values",
    Stage.DATA,
    "A categorical where one value dominates is usually a placeholder (unknown device, default site); treating it "
    "as an identity merges unrelated entities.",
    Severity.WARNING,
)
def dominant_values(ctx: Context) -> Outcome:
    """No categorical has a single value above `dominant_value_share` of rows."""
    findings: list[str] = []
    for column in ctx.data_config.categorical:
        top = ctx.events[column].value_counts(sort=True).row(0)
        share = top[1] / ctx.events.height
        if share > ctx.thresholds.dominant_value_share:
            findings.append(f"{column}={top[0]!r} on {share:.1%} of rows")
    if findings:
        return failed(f"{len(findings)} categoricals dominated by one value", findings)
    return passed(f"{len(ctx.data_config.categorical)} categoricals checked")


@check(
    "MLD008",
    "foreign-keys",
    Stage.DATA,
    "Events whose entity is missing get no entity features; a broken join must fail rather than become cold start.",
)
def foreign_keys(ctx: Context) -> Outcome:
    """Every foreign key value resolves in its entity table."""
    problems: list[str] = []
    for fk in ctx.data_config.foreign_keys:
        ref = ctx.reference_table(fk.ref_path)
        if ref[fk.ref_column].n_unique() != ref.height:
            problems.append(f"{fk.ref_path}:{fk.ref_column} is not unique")
        orphans = ctx.events.join(ref.select(fk.ref_column), left_on=fk.column, right_on=fk.ref_column, how="anti")
        if orphans.height:
            problems.append(f"{fk.column}: {orphans.height:,} rows without a match in {fk.ref_path}")
    if problems:
        return failed("referential integrity broken", problems)
    return passed(f"{len(ctx.data_config.foreign_keys)} foreign keys resolve")


@check(
    "MLD009",
    "event-after-creation",
    Stage.DATA,
    "An event before its entity existed means a corrupted timestamp or an entity attribute recorded after the fact.",
)
def event_after_creation(ctx: Context) -> Outcome:
    """Event time is not earlier than the referenced entity's creation date."""
    problems: list[str] = []
    for fk in [f for f in ctx.data_config.foreign_keys if f.ref_created_at]:
        created = ctx.reference_table(fk.ref_path).select(
            pl.col(fk.ref_column),
            pl.col(fk.ref_created_at or "").str.strptime(pl.Datetime, fk.ref_time_format).alias("__created"),
        )
        joined = ctx.events.select(fk.column, "__ts").join(created, left_on=fk.column, right_on=fk.ref_column)
        early = joined.filter(pl.col("__ts").dt.date() < pl.col("__created").dt.date())
        if early.height:
            problems.append(f"{fk.column}: {early.height:,} events before {fk.ref_created_at}")
    return failed("events precede entity creation", problems) if problems else passed("event times follow creation")


@check(
    "MLD010",
    "constant-columns",
    Stage.DATA,
    "A constant column carries no signal and usually signals a broken upstream join.",
    Severity.WARNING,
)
def constant_columns(ctx: Context) -> Outcome:
    """No declared column holds a single value."""
    present = [c for c in _expected_columns(ctx) if c in ctx.events.columns]
    constant = [c for c in present if ctx.events[c].n_unique() <= 1]
    return failed("constant columns", constant) if constant else passed("no constant columns")
