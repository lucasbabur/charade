from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from charade.data.load import load_characters
from charade.text import bakeoff, embed
from charade.text.embed import Provider
from tests.conftest import FIXTURES, TRAIN_END


def test_tfidf_embeddings_are_cached(tmp_path: Path) -> None:
    characters = load_characters(FIXTURES / "characters.csv")
    first = embed.raw_embeddings(characters, Provider.TFIDF, tmp_path)
    assert len(list(tmp_path.glob("tfidf_*.npy"))) == 1
    np.testing.assert_array_equal(first, embed.raw_embeddings(characters, Provider.TFIDF, tmp_path))


def test_pca_ignores_characters_published_after_training() -> None:
    characters = pl.DataFrame(
        {
            "character_id": ["a", "b", "c", "late"],
            "created_at": [datetime(2014, 1, 1)] * 3 + [datetime(2014, 10, 29)],
        }
    )
    base = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=np.float32)
    outlier = base.copy()
    outlier[3] = [1e6, -1e6]
    reduced = embed.reduce(characters, base, TRAIN_END, dims=1)["text_0"].to_numpy()[:3]
    shifted = embed.reduce(characters, outlier, TRAIN_END, dims=1)["text_0"].to_numpy()[:3]
    np.testing.assert_allclose(np.abs(reduced), np.abs(shifted), atol=1e-5)


def test_bakeoff_writes_reduced_vectors_and_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(embed, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(bakeoff, "MIN_IMPRESSIONS", 5)
    table = bakeoff.run([Provider.TFIDF], FIXTURES, tmp_path / "derived", tmp_path / "bakeoff.csv")
    assert table["genre_knn_acc"][0] > 0.9
    assert (tmp_path / "derived" / "text_tfidf.parquet").is_file()
    assert table["rho_ci_low"][0] <= table["residual_rho"][0] <= table["rho_ci_high"][0]
