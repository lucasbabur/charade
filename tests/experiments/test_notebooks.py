"""Every experiment notebook executes top to bottom in smoke mode (fixture data, tiny model).

Marked `notebooks` (slow): run with `uv run poe test-notebooks`; CI runs it as its own step.
"""

from pathlib import Path

import jupytext
import pytest
from nbclient import NotebookClient

from charade.analysis.experiment import SMOKE_ENV, TMP_ENV

ROOT = Path(__file__).parents[2]
NOTEBOOKS = sorted(ROOT.glob("experiments/E*/notebook.py"))


@pytest.fixture(scope="module")
def smoke_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("experiments")


@pytest.mark.notebooks
@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.parent.name)
def test_notebook_executes_in_smoke_mode(path: Path, smoke_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SMOKE_ENV, "1")
    monkeypatch.setenv(TMP_ENV, str(smoke_dir))
    monkeypatch.setenv("MPLBACKEND", "Agg")
    notebook = jupytext.read(path)
    NotebookClient(
        notebook, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}}
    ).execute()
    assert all(cell.get("execution_count") for cell in notebook.cells if cell.cell_type == "code")
