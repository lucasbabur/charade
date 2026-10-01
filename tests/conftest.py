from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from charade.analysis.experiment import smoke_settings
from charade.config import Settings
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


def fixture_settings(artifacts: Path) -> Settings:
    """Settings that point at the fixture data, a tiny model and a temporary artifact directory."""
    return smoke_settings(artifacts)


@pytest.fixture(scope="session")
def bundle(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    """A trained bundle on the fixture (tiny DCN), shared by OPE and serving tests."""
    from charade.models import pipeline  # noqa: PLC0415 - heavy import only for tests that need it

    root = tmp_path_factory.mktemp("bundle")
    settings = fixture_settings(root / "art")
    pipeline.run(settings, reports=root / "reports")
    return settings


@pytest.fixture(scope="session")
def sample_body(bundle: Settings) -> dict[str, object]:
    from charade.analysis.requests import creative_pool, request_from_row  # noqa: PLC0415
    from charade.models.dataset import build_frame  # noqa: PLC0415

    frame = build_frame(bundle, FIXTURES)
    test = frame.filter(pl.col("split") == "test")
    pool = creative_pool(frame).head(6).to_dicts()
    row = test.row(0, named=True)
    return request_from_row(row, pool, "req-1").model_dump(mode="json")
