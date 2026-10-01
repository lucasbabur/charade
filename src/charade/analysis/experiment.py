"""Shared setup for the experiment notebooks in `experiments/E*/notebook.py`.

Two modes:
- full (default): the real data and the shipped bundle; reports are written to `reports/`, the same
  files the `poe` tasks produce.
- smoke (`CHARADE_EXPERIMENT_SMOKE=1`): the test fixture, a tiny model and a temporary directory.
  CI executes every notebook this way, so a notebook cannot rot or depend on hidden state.
"""

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from charade.config import DcnConfig, GbdtConfig, ModelConfig, Settings, get_settings

SMOKE_ENV = "CHARADE_EXPERIMENT_SMOKE"
TMP_ENV = "CHARADE_EXPERIMENT_TMP"

SMOKE_MODEL = ModelConfig(
    groups=["context", "device", "ad", "character_meta", "user_history"],
    dcn=DcnConfig(
        embedding_dim=4, cross_layers=1, cross_rank=8, hidden=[16], max_epochs=2, batch_size=512, seeds=[0, 1, 2]
    ),
    gbdt=GbdtConfig(num_leaves=7, max_rounds=30, min_data_in_leaf=20),
)


@dataclass(frozen=True)
class Context:
    """Where an experiment reads data and writes results."""

    settings: Settings
    data_dir: Path
    reports: Path
    smoke: bool


def repo_root(start: Path | None = None) -> Path:
    """The directory holding pyproject.toml and src/charade."""
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "charade").is_dir():
            return candidate
    raise FileNotFoundError("run experiments from inside the repository")


def smoke_settings(artifacts: Path, root: Path | None = None) -> Settings:
    """Fixture data, tiny model, artifacts in `artifacts` (also used by the test suite)."""
    fixtures = (root or repo_root(Path(__file__))) / "tests" / "fixtures"
    return get_settings().model_copy(update={"data_dir": fixtures, "artifacts_dir": artifacts, "model": SMOKE_MODEL})


def setup() -> Context:
    """Change to the repository root and return the context for the current mode."""
    root = repo_root()
    os.chdir(root)
    if os.environ.get(SMOKE_ENV) == "1":
        tmp = Path(os.environ.get(TMP_ENV) or tempfile.mkdtemp(prefix="charade-exp-"))
        settings = smoke_settings(tmp / "artifacts", root)
        return Context(settings, settings.data_dir, tmp / "reports", smoke=True)
    settings = get_settings()
    return Context(settings, settings.data_dir, Path("reports"), smoke=False)


def ensure_bundle(ctx: Context) -> None:
    """Make sure a trained bundle exists: train a tiny one in smoke mode, demand E004 otherwise."""
    if (ctx.settings.artifacts_dir / "model.onnx").is_file():
        return
    if not ctx.smoke:
        raise FileNotFoundError("no model bundle: run experiments/E004 (or `uv run poe train`) first")
    from charade.models import pipeline  # noqa: PLC0415 - heavy import only when a bundle is missing

    pipeline.run(ctx.settings, reports=ctx.reports / "models")
