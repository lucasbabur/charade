"""Embed character descriptions with several providers and reduce them to a few dimensions.

Raw embeddings are cached under `artifacts/cache/text/` keyed by provider; only the reduced vectors
(`data/derived/text_<provider>.parquet`, ~0.6 MB each) are committed, so every result reproduces
without API keys. PCA is fitted on characters published by the end of training (MLL003).
"""

import hashlib
import os
from collections.abc import Callable
from datetime import datetime
from enum import StrEnum
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from tenacity import retry, stop_after_attempt, wait_exponential

from charade.features.spec import TEXT_PREFIX

DERIVED_DIR = Path("data/derived")
CACHE_DIR = Path("artifacts/cache/text")
DIMS = 16

type Matrix = npt.NDArray[np.float32]


class Provider(StrEnum):
    """Embedding providers in the bake-off."""

    TFIDF = "tfidf"
    QWEN3 = "qwen3_0_6b"
    OPENAI = "openai_3_large"


def _tfidf(texts: list[str]) -> Matrix:
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit_transform(texts)
    return TruncatedSVD(n_components=128, random_state=0).fit_transform(tfidf).astype(np.float32)


def _qwen3(texts: list[str]) -> Matrix:
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415 - 1 GB model, load on demand

    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cuda" if _has_cuda() else "cpu")
    return model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)


def _has_cuda() -> bool:
    import torch  # noqa: PLC0415

    return torch.cuda.is_available()


def _openai(texts: list[str]) -> Matrix:
    from openai import OpenAI  # noqa: PLC0415

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=30))
    def batch(chunk: list[str]) -> list[list[float]]:
        response = client.embeddings.create(model="text-embedding-3-large", input=chunk, dimensions=1024)
        return [item.embedding for item in response.data]

    vectors = [v for start in range(0, len(texts), 256) for v in batch(texts[start : start + 256])]
    return np.asarray(vectors, dtype=np.float32)


EMBEDDERS: dict[Provider, Callable[[list[str]], Matrix]] = {
    Provider.TFIDF: _tfidf,
    Provider.QWEN3: _qwen3,
    Provider.OPENAI: _openai,
}


def raw_embeddings(characters: pl.DataFrame, provider: Provider, cache_dir: Path = CACHE_DIR) -> Matrix:
    """Full-dimensional embeddings in `characters` row order, cached by provider and text hash."""
    texts = characters["character_description"].to_list()
    digest = hashlib.sha256("\n".join(texts).encode()).hexdigest()[:16]
    path = cache_dir / f"{provider}_{digest}.npy"
    if path.is_file():
        return np.load(path)
    vectors = EMBEDDERS[provider](texts)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(path, vectors)
    return vectors


def reduce(characters: pl.DataFrame, vectors: Matrix, fit_until: datetime, dims: int = DIMS) -> pl.DataFrame:
    """PCA to `dims`, fitted on characters created by `fit_until`; returns `character_id, text_0..`."""
    fit_rows = (characters["created_at"] <= fit_until).to_numpy()
    pca = PCA(n_components=dims, random_state=0).fit(vectors[fit_rows])
    reduced = pca.transform(vectors).astype(np.float32)
    columns = {f"{TEXT_PREFIX}{i}": reduced[:, i] for i in range(dims)}
    return pl.DataFrame({"character_id": characters["character_id"], **columns})


def derived_path(provider: Provider, derived_dir: Path = DERIVED_DIR) -> Path:
    """Committed reduced vectors for a provider."""
    return derived_dir / f"text_{provider}.parquet"
