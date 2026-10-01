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


def run(variants: tuple[Variant, ...] = VARIANTS, out: Path = OUT, data_dir: Path | None = None) -> pl.DataFrame:
    """Train every variant and write reports/models/ablations.csv."""
    settings = get_settings()
    reference: np.ndarray | None = None
    rows: list[dict[str, object]] = []
    cache: dict[tuple[frozenset[Group], Provider | None], Prepared] = {}
    for variant in variants:
        groups = ALL_GROUPS - variant.drop
        key = (groups, variant.text)
        if key not in cache:
            cache[key] = load_prepared(settings, data_dir or settings.data_dir, set(groups), variant.text)
        prep = cache[key]
        cfg = DcnConfig.model_validate({**settings.model.dcn.model_dump(), **variant.dcn_changes})
        ensemble, _ = train_dcn_ensemble(prep, cfg, settings.model.dcn.seeds)
        y = prep.y["val"]
        p = 1 / (1 + np.exp(-logits(ensemble, prep, "val")))
        loss = logloss_rows(y, p)
        if reference is None:
            reference = loss
        blocks = prep.frame.filter(pl.col("split") == "val")["ts"].dt.hour().to_numpy()
        delta, low, high = paired_bootstrap(loss - reference, blocks)
        rows.append(
            {
                "variant": variant.name,
                "val_ne": normalized_entropy(y, p),
                "delta_logloss": delta,
                "ci_low": low,
                "ci_high": high,
            }
        )
    table = pl.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    table.write_csv(out / "ablations.csv")
    return table


if __name__ == "__main__":
    print(run())
