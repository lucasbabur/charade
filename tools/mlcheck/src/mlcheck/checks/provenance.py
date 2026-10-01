"""Reproducibility and leakage checks on the run manifest, splits and leakage probes."""

import hashlib
from collections import Counter
from itertools import pairwise
from pathlib import Path

import polars as pl

from mlcheck.context import Context
from mlcheck.registry import check
from mlcheck.result import Outcome, Severity, Stage, failed, passed

_CHUNK = 1 << 20


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


@check(
    "MLR001",
    "manifest-valid",
    Stage.REPRO,
    "Without data hash, code version, config hash, seeds and windows a model cannot be rebuilt or audited.",
)
def manifest_valid(ctx: Context) -> Outcome:
    """manifest.json exists and satisfies the RunManifest contract."""
    manifest = ctx.manifest
    missing = [s for s in ctx.config.split_order if s not in manifest.windows]
    if missing:
        return failed("manifest lacks split windows", missing)
    return passed(f"run {manifest.run_id} at {manifest.git_sha[:8]}, {len(manifest.fitted_artifacts)} fitted artifacts")


@check(
    "MLR002",
    "data-hash-matches",
    Stage.REPRO,
    "Reported metrics are only meaningful for the exact bytes the model was trained on.",
)
def data_hash_matches(ctx: Context) -> Outcome:
    """Every file hashed in the manifest still has the same sha256."""
    problems: list[str] = []
    for relative, expected in ctx.manifest.data_sha256.items():
        path = ctx.root / relative
        if not path.is_file():
            problems.append(f"{relative}: missing")
        elif (actual := _sha256(path)) != expected:
            problems.append(f"{relative}: {actual[:12]} != manifest {expected[:12]}")
    if problems:
        return failed("data changed since training", problems)
    return passed(f"{len(ctx.manifest.data_sha256)} data files match")


@check(
    "MLR003",
    "clean-git-tree",
    Stage.REPRO,
    "A model trained from uncommitted code cannot be traced to a reviewable commit.",
)
def clean_git_tree(ctx: Context) -> Outcome:
    """The run was trained from a clean working tree."""
    if ctx.manifest.git_dirty:
        return failed(f"trained from a dirty tree at {ctx.manifest.git_sha[:8]}")
    return passed(f"trained at clean commit {ctx.manifest.git_sha[:8]}")


@check(
    "MLR004",
    "seed-count",
    Stage.REPRO,
    "Neural nets vary across seeds by about as much as many feature changes; one seed cannot separate the two.",
)
def seed_count(ctx: Context) -> Outcome:
    """The primary model was trained with at least `min_seeds` seeds."""
    seeds = ctx.manifest.model_seeds.get(ctx.config.primary_model, [])
    distinct = len(set(seeds))
    if distinct < ctx.thresholds.min_seeds:
        return failed(f"{ctx.config.primary_model}: {distinct} distinct seeds, need {ctx.thresholds.min_seeds}")
    return passed(f"{ctx.config.primary_model}: {distinct} seeds")


@check("MLL001", "split-disjoint", Stage.LEAKAGE, "An event in two splits is evaluated on data it was trained on.")
def split_disjoint(ctx: Context) -> Outcome:
    """No id belongs to more than one split, and every configured split is non-empty."""
    splits = ctx.splits
    sizes = dict(splits.group_by("split").len().iter_rows())
    empty = [s for s in ctx.config.split_order if not sizes.get(s)]
    if empty:
        return failed("empty or missing splits", empty)
    shared = splits.group_by("id").agg(pl.col("split").n_unique().alias("n")).filter(pl.col("n") > 1).height
    if shared:
        return failed(f"{shared:,} ids appear in more than one split")
    return passed(", ".join(f"{s}={sizes[s]:,}" for s in ctx.config.split_order))


@check(
    "MLL002",
    "split-temporal-order",
    Stage.LEAKAGE,
    "Production predicts the future from the past; each split must start strictly after the previous one ends, and "
    "match the manifest windows.",
)
def split_temporal_order(ctx: Context) -> Outcome:
    """Splits are strictly time-ordered and lie inside the manifest windows."""
    bounds = {
        r["split"]: (r["start"], r["end"])
        for r in ctx.splits.group_by("split")
        .agg(pl.col("ts").min().alias("start"), pl.col("ts").max().alias("end"))
        .iter_rows(named=True)
    }
    order = ctx.config.split_order
    problems = [
        f"{earlier} ends {bounds[earlier][1]} >= {later} starts {bounds[later][0]}"
        for earlier, later in pairwise(order)
        if bounds[earlier][1] >= bounds[later][0]
    ]
    for name in order:
        window = ctx.manifest.windows[name]
        start, end = bounds[name]
        if start < window.start or end > window.end:
            problems.append(f"{name} spans {start}..{end}, outside manifest window {window.start}..{window.end}")
    return failed("splits overlap in time", problems) if problems else passed(" < ".join(order))


@check(
    "MLL003",
    "fit-window",
    Stage.LEAKAGE,
    "Every fitted object (vocabulary, encoder, prior, scaler, model, calibrator) must be fitted only on data that "
    "precedes the holdout, inside the split it declares.",
)
def fit_window(ctx: Context) -> Outcome:
    """No fitted artifact uses holdout data or data after its declared split."""
    holdout = ctx.config.holdout_split
    holdout_start = ctx.manifest.windows[holdout].start
    problems: list[str] = []
    for artifact in ctx.manifest.fitted_artifacts:
        window = ctx.manifest.windows.get(artifact.fit_split)
        if artifact.fit_split == holdout or window is None:
            problems.append(f"{artifact.name}: fitted on '{artifact.fit_split}'")
        elif artifact.fit_end > window.end or artifact.fit_end >= holdout_start:
            problems.append(f"{artifact.name}: fit_end {artifact.fit_end} beyond {artifact.fit_split} window")
    if problems:
        return failed("artifacts fitted on data they must not see", problems)
    return passed(f"{len(ctx.manifest.fitted_artifacts)} artifacts fitted before {holdout_start}")


@check(
    "MLL004",
    "shuffled-label-auc",
    Stage.LEAKAGE,
    "Retraining on shuffled labels must give chance-level AUC; anything else means the pipeline leaks the label.",
)
def shuffled_label_auc(ctx: Context) -> Outcome:
    """Validation AUC of a model trained on permuted labels is within the chance band."""
    auc = ctx.leakage.shuffled_label_auc
    low, high = ctx.thresholds.shuffled_auc_low, ctx.thresholds.shuffled_auc_high
    if not low <= auc <= high:
        return failed(f"shuffled-label AUC {auc:.4f} outside [{low}, {high}]")
    return passed(f"shuffled-label AUC {auc:.4f}")


@check(
    "MLL005",
    "univariate-feature-auc",
    Stage.LEAKAGE,
    "No single feature should predict clicks almost perfectly; one that does is usually derived from the label.",
)
def univariate_feature_auc(ctx: Context) -> Outcome:
    """Each feature's standalone validation AUC is below `max_univariate_auc`."""
    limit = ctx.thresholds.max_univariate_auc
    suspicious = {f: a for f, a in ctx.leakage.feature_auc.items() if max(a, 1 - a) > limit}
    if suspicious:
        return failed("features suspiciously predictive alone", [f"{f}: {a:.4f}" for f, a in suspicious.items()])
    if not ctx.leakage.feature_auc:
        return failed("leakage.json has no per-feature AUCs")
    top = max(ctx.leakage.feature_auc.items(), key=lambda kv: max(kv[1], 1 - kv[1]))
    return passed(f"{len(ctx.leakage.feature_auc)} features below {limit}; strongest {top[0]}={top[1]:.4f}")


@check(
    "MLL006",
    "holdout-touched-once",
    Stage.LEAKAGE,
    "Evaluating repeatedly on the holdout and picking the best turns it into a validation set; each model version "
    "gets one holdout evaluation.",
)
def holdout_touched_once(ctx: Context) -> Outcome:
    """Each model version appears at most once on the holdout in the evaluation ledger, and the current run once."""
    holdout = ctx.config.holdout_split
    counts = Counter(e.model_version for e in ctx.ledger if e.split == holdout)
    repeated = [f"{version}: {n} holdout evaluations" for version, n in counts.items() if n > 1]
    if repeated:
        return failed("holdout evaluated more than once", repeated)
    current = [e for e in ctx.ledger if e.split == holdout and e.run_id == ctx.manifest.run_id]
    if not current:
        return failed(f"run {ctx.manifest.run_id} has no holdout evaluation in the ledger")
    return passed(f"{len(counts)} model versions, one holdout evaluation each")


@check(
    "MLL007",
    "adversarial-validation",
    Stage.LEAKAGE,
    "A classifier that separates train from holdout rows with high AUC means strong covariate shift; offline "
    "metrics on that holdout will not transfer.",
    Severity.WARNING,
)
def adversarial_validation(ctx: Context) -> Outcome:
    """Train-vs-holdout classifier AUC is below `max_adversarial_auc`."""
    auc = ctx.leakage.adversarial_auc
    if auc > ctx.thresholds.max_adversarial_auc:
        return failed(f"adversarial AUC {auc:.4f} > {ctx.thresholds.max_adversarial_auc}")
    return passed(f"adversarial AUC {auc:.4f}")
