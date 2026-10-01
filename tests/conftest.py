from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from charade.config import DcnConfig, GbdtConfig, ModelConfig, Settings, get_settings
from charade.data.load import load_joined
from charade.data.split import assign_split
from charade.features.counters import offline_counters
from charade.features.derive import derive, user_proxy

FIXTURES = Path(__file__).parent / "fixtures"
TRAIN_END = datetime(2014, 10, 27, 23)
VAL_END = datetime(2014, 10, 28, 23)


@pytest.fixture(scope="session")
def joined() -> pl.DataFrame:
    return load_joined(FIXTURES)


@pytest.fixture(scope="session")
def features(joined: pl.DataFrame) -> pl.DataFrame:
    frame = derive(offline_counters(joined.with_columns(user_proxy())))
    return assign_split(frame, TRAIN_END, VAL_END)


TINY_MODEL = ModelConfig(
    groups=["context", "device", "ad", "character_meta", "user_history"],
    dcn=DcnConfig(
        embedding_dim=4, cross_layers=1, cross_rank=8, hidden=[16], max_epochs=2, batch_size=512, seeds=[0, 1, 2]
    ),
    gbdt=GbdtConfig(num_leaves=7, max_rounds=30, min_data_in_leaf=20),
)


def fixture_settings(artifacts: Path) -> Settings:
    """Settings that point at the fixture data and a temporary artifact directory."""
    return get_settings().model_copy(update={"data_dir": FIXTURES, "artifacts_dir": artifacts, "model": TINY_MODEL})


@pytest.fixture(scope="session")
def bundle(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    """A trained bundle on the fixture (tiny DCN), shared by OPE and serving tests."""
    from charade.models import pipeline  # noqa: PLC0415 - heavy import only for tests that need it

    root = tmp_path_factory.mktemp("bundle")
    settings = fixture_settings(root / "art")
    pipeline.run(settings, reports=root / "reports")
    return settings
