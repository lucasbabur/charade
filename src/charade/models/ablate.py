"""Ablations on validation (`uv run poe ablate`): which feature groups earn their place?

Each variant is a seed ensemble with every non-text feature group ("all features") except for
one change. Its validation log loss is compared to that reference with a paired hour-block bootstrap; a positive
delta means the variant is worse, so the removed group helps. Raw (uncalibrated) logits are
compared, because the calibrator itself is fitted on validation. Test is never used here.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from charade.config import DcnConfig, get_settings
from charade.evaluation.metrics import logloss_rows, normalized_entropy, paired_bootstrap
from charade.features.spec import Group
from charade.models.core import Prepared, load_prepared, logits, train_dcn_ensemble
from charade.text.embed import Provider

OUT = Path("reports/models")


@dataclass(frozen=True)
class Variant:
    """One ablation: groups removed, text added, or a config change."""

    name: str
    drop: frozenset[Group] = frozenset()
    text: Provider | None = None
    dcn_changes: dict[str, object] = field(default_factory=dict[str, object])


ALL_GROUPS = frozenset(set(Group) - {Group.TEXT})

VARIANTS = (
    Variant("all features"),
    Variant("- character_meta", drop=frozenset({Group.CHARACTER_META})),
    Variant("- character_id", drop=frozenset({Group.CHARACTER_ID})),
    Variant("- all character features", drop=frozenset({Group.CHARACTER_META, Group.CHARACTER_ID})),
    Variant("- user_history", drop=frozenset({Group.USER_HISTORY})),
    Variant("- conversation", drop=frozenset({Group.CONVERSATION})),
    Variant("- device", drop=frozenset({Group.DEVICE})),
    Variant("+ text (qwen3)", text=Provider.QWEN3),
    Variant("+ text (tfidf)", text=Provider.TFIDF),
    Variant("no character-id dropout", dcn_changes={"character_id_dropout": 0.0}),
    Variant("shipped: - character_id - conversation", drop=frozenset({Group.CHARACTER_ID, Group.CONVERSATION})),
)


SEED_OFFSET = 100
"""The second, independent seed set is the configured seeds + 100."""


def _verdict(lows: list[float], highs: list[float]) -> str:
    if all(low > 0 for low in lows):
        return "worse than reference"
    if all(high < 0 for high in highs):
        return "variant better"
    return "within noise"


def run(variants: tuple[Variant, ...] = VARIANTS, out: Path = OUT, data_dir: Path | None = None) -> pl.DataFrame:
    """Train every variant with two independent seed sets; write reports/models/ablations.csv.

    The hour-block bootstrap captures data noise, not training randomness, so each comparison is
    repeated with a second seed set (both reference and variant retrained). A variant's effect is
    only called real when both seed sets give intervals on the same side of zero; the reported
    interval is the union of the two (conservative).
    """
    settings = get_settings()
    seed_sets = [settings.model.dcn.seeds, [s + SEED_OFFSET for s in settings.model.dcn.seeds]]
    references: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    cache: dict[tuple[frozenset[Group], Provider | None], Prepared] = {}
    for variant in variants:
        groups = ALL_GROUPS - variant.drop
        key = (groups, variant.text)
        if key not in cache:
            cache[key] = load_prepared(settings, data_dir or settings.data_dir, set(groups), variant.text)
        prep = cache[key]
        cfg = DcnConfig.model_validate({**settings.model.dcn.model_dump(), **variant.dcn_changes})
        y = prep.y["val"]
        blocks = prep.frame.filter(pl.col("split") == "val")["ts"].dt.hour().to_numpy()
        deltas, lows, highs, nes = [], [], [], []
        for index, seeds in enumerate(seed_sets):
            ensemble, _ = train_dcn_ensemble(prep, cfg, seeds)
            p = 1 / (1 + np.exp(-logits(ensemble, prep, "val")))
            loss = logloss_rows(y, p)
            if len(references) <= index:
                references.append(loss)
            delta, low, high = paired_bootstrap(loss - references[index], blocks)
            deltas.append(delta)
            lows.append(low)
            highs.append(high)
            nes.append(normalized_entropy(y, p))
        rows.append(
            {
                "variant": variant.name,
                "val_ne": float(np.mean(nes)),
                "delta_logloss": float(np.mean(deltas)),
                "ci_low": min(lows),
                "ci_high": max(highs),
                "delta_seed_set_a": deltas[0],
                "delta_seed_set_b": deltas[1],
                "verdict": "reference"
                if not variant.drop and not variant.text and not variant.dcn_changes
                else _verdict(lows, highs),
            }
        )
    table = pl.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    table.write_csv(out / "ablations.csv")
    return table


if __name__ == "__main__":
    print(run())
