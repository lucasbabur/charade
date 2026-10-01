"""Execute experiment notebooks: `notebook.py` (source) -> executed, paired `notebook.ipynb`.

`uv run poe experiments [paths]` runs on the full data; with `CHARADE_EXPERIMENT_SMOKE=1` it uses the
fixture. Uses nbclient directly (the same executor the CI test uses), so no nbconvert dependency.
"""

import sys
from pathlib import Path

import jupytext
from nbclient import NotebookClient

FORMATS = "ipynb,py:percent"


def execute(source: Path, timeout: int = 3600) -> Path:
    """Run `source` top to bottom and write the paired, executed notebook next to it."""
    notebook = jupytext.read(source)
    notebook.metadata.setdefault("jupytext", {})["formats"] = FORMATS
    notebook.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    NotebookClient(
        notebook, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(source.parent)}}
    ).execute()
    target = source.with_suffix(".ipynb")
    jupytext.write(notebook, target)
    jupytext.write(notebook, source, fmt="py:percent")
    return target


def main(paths: list[str]) -> None:
    """Execute the given notebooks, or every experiment in id order."""
    sources = [Path(p) for p in paths] or sorted(Path("experiments").glob("E*/notebook.py"))
    for source in sources:
        print(f"executing {source}", flush=True)
        execute(source)


if __name__ == "__main__":
    main(sys.argv[1:])
