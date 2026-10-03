"""Model comparisons across data volumes (`python -m charade.analysis.fullscale <train.gz>`).

The shipped bundle needs the synthetic character layer, which the full Kaggle Avazu train file
(~40M rows, the same ten days) does not have. So this compares like with like: the same pipeline
(causal user counters, the same temporal split and test window, the same tuned DCN-v2 and LightGBM)
with the character and conversation groups removed, trained once on the 1M-row sample and once on
the full file. It answers whether the model family and its ranking hold up with more data; it says
nothing about the character features.

The comparison is not to the Kaggle leaderboard: those numbers come from random splits and often
same-hour counters, which this project avoids (docs/03-models-evaluation.md).
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from charade.config import Settings, get_settings
from charade.data.load import IMPRESSION_COLUMNS, parse_hour
from charade.data.split import assign_split
from charade.evaluation.metrics import summary
from charade.features.counters import offline_counters
from charade.features.derive import derive, user_proxy
from charade.features.spec import Group
from charade.models.core import prepare, train_dcn
from charade.models.gbdt import predict_gbdt, train_gbdt
from charade.models.trainer import predict_logits

OUT = Path("reports/models/fullscale.json")
TEST_END = datetime(2014, 10, 30, 5)
USER_SHARE = 4
SMALL_SHARE = 36
"""1 in 36 users gives ~0.74M training rows, about the 1M sample's training size."""
"""The full 40M rows need ~150 GB in this in-memory pipeline; every impression of 1 in 4 users (~9M rows,
9x the sample) fits in ~25 GB and keeps each kept user's history, and so the counters, exact."""
GROUPS = {Group.CONTEXT, Group.DEVICE, Group.AD, Group.USER_HISTORY}
NEUTRAL = {
    "character_id": pl.lit("none"),
    "genre": pl.lit("none"),
    "safety_tier": pl.lit("sfw"),
    "creator_type": pl.lit("none"),
    "num_interactions": pl.lit(0, pl.Int64),
    "created_at": pl.lit("2014-01-01").str.strptime(pl.Datetime("us"), "%Y-%m-%d"),
    "conversation_turn": pl.lit(1, pl.Int32),
    "session_msg_count": pl.lit(1, pl.Int32),
}
"""Constants for the character and conversation inputs `derive` expects; their groups are not used."""
AVAZU_COLUMNS = [c for c in IMPRESSION_COLUMNS if c not in {"character_id", "conversation_turn", "session_msg_count"}]


def _frame(path: Path, settings: Settings, user_share: int = 1) -> pl.DataFrame:
    """Avazu columns up to the test window, keeping every impression of 1 in `user_share` users.

    Sampling users, not rows, keeps each kept user's history complete, so the counters stay exact.
    """
    frame = (
        pl.scan_csv(path, schema_overrides={c: IMPRESSION_COLUMNS[c] for c in AVAZU_COLUMNS})
        .select(AVAZU_COLUMNS)
        .with_columns(parse_hour(pl.col("hour")).alias("ts"), user_proxy())
        .filter((pl.col("ts") <= settings.test_end) & (pl.col("user").hash(seed=settings.seed) % user_share == 0))
        .with_columns(**NEUTRAL)
        .collect(engine="streaming")
    )
    frame = offline_counters(frame)
    return assign_split(derive(frame), settings.train_end, settings.val_end, settings.test_end).sort(["ts", "id"])


def _evaluate(name: str, frame: pl.DataFrame, settings: Settings) -> dict[str, object]:
    started = time.monotonic()
    prep = prepare(frame, GROUPS, text=False)
    seed = settings.model.dcn.seeds[0]
    dcn = train_dcn(prep, settings.model.dcn, seed)
    gbdt = train_gbdt(
        prep.spec, prep.x["train"], prep.y["train"], prep.x["val"], prep.y["val"], settings.model.gbdt, seed
    )
    y = prep.y["test"]
    p_dcn = 1 / (1 + np.exp(-predict_logits(dcn.model, prep.x["test"])))
    p_gbdt = predict_gbdt(gbdt, prep.x["test"])
    rows = {s: len(prep.y[s]) for s in ("train", "val", "test")}
    return {
        "data": name,
        "rows": rows,
        "dcn_v2": summary(y, p_dcn),
        "lightgbm": summary(y, p_gbdt),
        "dcn_steps": dcn.best_step,
        "lightgbm_rounds": gbdt.best_iteration,
        "seconds": time.monotonic() - started,
    }


def run(
    full_path: Path,
    settings: Settings | None = None,
    out: Path = OUT,
    user_share: int = USER_SHARE,
    small_share: int = SMALL_SHARE,
) -> dict[str, object]:
    """Volume effect on one shared test set, plus the sample baseline on the sample's own test set.

    From the full file (1 in `user_share` users), two models are trained: one on 1 in `small_share`
    users' train and validation rows (about the sample's size), one on all of them. Both are scored on
    the same test rows. Both training and validation volume change; this is a one-seed descriptive
    comparison, not an isolated training-volume effect. `small_share` must be a multiple
    of `user_share` so the small run's users are a subset of the large run's.
    """
    if small_share % user_share:
        raise ValueError("small_share must be a multiple of user_share")
    # The sample ends at 2014-10-30 05:00; cut the full file at the same hour so both tests cover the same window.
    settings = (settings or get_settings()).model_copy(update={"test_end": TEST_END})
    full = _frame(full_path, settings, user_share)
    small = full.filter((pl.col("split") == "test") | (pl.col("user").hash(seed=settings.seed) % small_share == 0))
    report: dict[str, object] = {
        "groups": sorted(str(g) for g in GROUPS),
        "seed": settings.model.dcn.seeds[0],
        "user_share": user_share,
        "small_share": small_share,
        "sample_baseline": _evaluate(
            "1M sample, no character layer (sample test set)",
            _frame(settings.data_dir / "impressions.csv", settings),
            settings,
        ),
        "shared_test": [
            _evaluate(f"full Avazu, 1 in {small_share} users", small, settings),
            _evaluate(f"full Avazu, 1 in {user_share} users", full, settings),
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    return report


if __name__ == "__main__":
    print(json.dumps(run(Path(sys.argv[1])), indent=2, default=str))
