"""The modelling table: raw data -> user proxy -> causal counters -> text vectors -> derived columns -> split."""

from pathlib import Path

import polars as pl

from charade.config import Settings
from charade.data.load import load_joined
from charade.data.split import assign_split
from charade.features.counters import offline_counters
from charade.features.derive import derive, user_proxy
from charade.text.embed import Provider, derived_path


def build_frame(
    settings: Settings, data_dir: Path, text_provider: Provider | None = None, derived_dir: Path | None = None
) -> pl.DataFrame:
    """Every derived column for every impression, with `split` from the given settings' windows."""
    frame = offline_counters(load_joined(data_dir).with_columns(user_proxy()))
    if text_provider is not None:
        path = derived_path(text_provider) if derived_dir is None else derived_path(text_provider, derived_dir)
        frame = frame.join(pl.read_parquet(path), on="character_id", how="left")
    return assign_split(derive(frame), settings.train_end, settings.val_end, settings.test_end).sort(["ts", "id"])
