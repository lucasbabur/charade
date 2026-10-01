"""Documentation guards: frontmatter, freshness, links and commands (cheap checks that catch rot)."""

import json
import re
import subprocess
import tomllib
from datetime import date
from pathlib import Path

import jupytext
import pytest

ROOT = Path(__file__).parents[2]
DOCS = sorted([*ROOT.glob("docs/*.md"), *ROOT.glob("docs/adr/*.md")])
EXPERIMENTS = sorted(ROOT.glob("experiments/E*/README.md"))
EXPERIMENT_FIELDS = ("id", "title", "hypotheses", "status", "conclusion", "created-at", "updated-at")
LINKED = [
    *DOCS,
    *EXPERIMENTS,
    ROOT / "README.md",
    ROOT / "AGENTS.md",
    ROOT / "tools/mlcheck/AGENTS.md",
    ROOT / "tools/mlcheck/README.md",
]
FIELDS = ("title", "created-at", "updated-at")
LINK = re.compile(r"\]\(([^)\s]+)\)")
POE = re.compile(r"\bpoe\s+([a-z][a-z0-9-]*)")


def _frontmatter(path: Path) -> dict[str, str]:
    text = path.read_text()
    if not text.startswith("---\n"):
        return {}
    block = text[4 : text.index("\n---\n", 4)]
    return {k.strip(): v.strip().strip('"') for k, _, v in (line.partition(":") for line in block.splitlines())}


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()  # noqa: S603, S607


def _slug(heading: str) -> str:
    """GitHub anchor for a markdown heading."""
    text = re.sub(r"[^\w\- ]", "", heading.strip().lower())
    return text.replace(" ", "-")


def _anchors(path: Path) -> set[str]:
    return {_slug(m.group(1)) for m in re.finditer(r"^#+ (.+)$", path.read_text(), re.MULTILINE)}


@pytest.mark.parametrize("path", DOCS, ids=lambda p: str(p.relative_to(ROOT)))
def test_frontmatter_has_exactly_title_and_dates(path: Path) -> None:
    meta = _frontmatter(path)
    assert tuple(meta) == FIELDS, f"{path.name}: frontmatter must be exactly {FIELDS}"
    assert meta["title"]
    created, updated = date.fromisoformat(meta["created-at"]), date.fromisoformat(meta["updated-at"])
    assert created <= updated


@pytest.mark.parametrize("path", [*DOCS, *EXPERIMENTS], ids=lambda p: str(p.relative_to(ROOT)))
def test_updated_at_is_bumped_when_the_doc_changes(path: Path) -> None:
    """`updated-at` must be no older than the last commit that touched the doc (author date survives rebases).

    For an experiment, any change in its folder (README or notebook) counts.
    """
    relative = str((path.parent if path in EXPERIMENTS else path).relative_to(ROOT))
    if _git("rev-parse", "--is-shallow-repository") == "true":
        pytest.skip("shallow clone: history unavailable (CI checks out with fetch-depth 0)")
    last = _git("log", "-1", "--format=%as", "--", relative)
    updated = date.fromisoformat(_frontmatter(path)["updated-at"])
    if last:
        assert updated >= date.fromisoformat(last), f"{relative}: changed on {last}, updated-at says {updated}"
    if _git("status", "--porcelain", "--", relative):
        assert updated >= date.today(), f"{relative}: uncommitted edits, bump updated-at to {date.today()}"


@pytest.mark.parametrize("path", LINKED, ids=lambda p: str(p.relative_to(ROOT)))
def test_relative_links_resolve(path: Path) -> None:
    broken: list[str] = []
    for target in LINK.findall(path.read_text()):
        if re.match(r"^[a-z]+:", target):
            continue
        file_part, _, anchor = target.partition("#")
        resolved = (path.parent / file_part).resolve() if file_part else path
        if not resolved.exists() or (anchor and resolved.suffix == ".md" and anchor not in _anchors(resolved)):
            broken.append(target)
    assert not broken, f"{path.name}: broken links {broken}"


def test_documented_poe_tasks_exist() -> None:
    tasks = set(tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["poe"]["tasks"])
    missing = {
        f"{path.relative_to(ROOT)}: poe {task}"
        for path in LINKED
        for task in POE.findall(path.read_text())
        if task not in tasks
    }
    assert not missing, sorted(missing)


def _hypothesis_rows() -> dict[str, str]:
    text = (ROOT / "docs/hypotheses.md").read_text()
    return {m.group(1): m.group(0) for m in re.finditer(r"^\| (H\d+) \|.*$", text, re.MULTILINE)}


def _ids(value: str) -> list[str]:
    return [item.strip() for item in value.strip("[]").split(",") if item.strip()]


@pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda p: p.parent.name)
def test_experiment_metadata(path: Path) -> None:
    meta = _frontmatter(path)
    assert tuple(meta) == EXPERIMENT_FIELDS, f"{path.parent.name}: frontmatter must be exactly {EXPERIMENT_FIELDS}"
    assert path.parent.name.startswith(meta["id"] + "-")
    assert meta["status"] in {"planned", "running", "concluded"}
    assert set(_ids(meta["hypotheses"])) <= set(_hypothesis_rows()), "unknown hypothesis id"
    if meta["status"] == "concluded":
        assert meta["conclusion"], "a concluded experiment states its conclusion"


@pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda p: p.parent.name)
def test_notebook_is_paired_and_executed(path: Path) -> None:
    source, notebook = path.parent / "notebook.py", path.parent / "notebook.ipynb"
    paired = jupytext.writes(jupytext.read(notebook), fmt="py:percent")
    assert paired == source.read_text(), "notebook.ipynb and notebook.py differ: run `jupytext --sync`"
    if _frontmatter(path)["status"] == "concluded":
        cells = json.loads(notebook.read_text())["cells"]
        assert all(c.get("execution_count") for c in cells if c["cell_type"] == "code"), (
            "concluded notebook is not executed"
        )


def test_hypotheses_and_experiments_reference_each_other() -> None:
    """The hypotheses table links exactly the experiments whose frontmatter names that hypothesis."""
    rows = _hypothesis_rows()
    declared: dict[str, set[str]] = {h: set() for h in rows}
    for path in EXPERIMENTS:
        meta = _frontmatter(path)
        for hypothesis in _ids(meta["hypotheses"]):
            declared[hypothesis].add(meta["id"])
    linked = {h: set(re.findall(r"\[(E\d+)\]\(", row)) for h, row in rows.items()}
    assert linked == declared
    assert all(declared.values()), "every hypothesis has at least one experiment"


def test_every_experiment_is_indexed() -> None:
    index = (ROOT / "docs/index.md").read_text()
    assert all(f"experiments/{p.parent.name}/README.md" in index for p in EXPERIMENTS)


def test_summary_gate_line_matches_mlcheck_on_the_current_bundle() -> None:
    """The summary's gate counts are not hand-typed truth: compare them with a real mlcheck run."""
    if not (ROOT / "artifacts/current/manifest.json").is_file() or not (ROOT / "impressions.csv").is_file():
        pytest.skip("no trained bundle and raw data here (CI); checked wherever a bundle exists")
    out = subprocess.run(["mlcheck", "."], cwd=ROOT, capture_output=True, text=True, check=False).stdout  # noqa: S607
    totals = re.search(r"(\d+) checks: (\d+) pass, (\d+) fail, (\d+) warn", out)
    assert totals
    claimed = re.search(r"(\d+) checks, (\d+) pass, (\d+) fail, (\d+) warn", (ROOT / "docs/00-summary.md").read_text())
    assert claimed
    assert claimed.groups() == totals.groups(), f"summary says {claimed.groups()}, mlcheck says {totals.groups()}"
