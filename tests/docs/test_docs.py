"""Documentation guards: frontmatter, freshness, links and commands (cheap checks that catch rot)."""

import re
import subprocess
import tomllib
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
DOCS = sorted([*ROOT.glob("docs/*.md"), *ROOT.glob("docs/adr/*.md")])
LINKED = [
    *DOCS,
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


@pytest.mark.parametrize("path", DOCS, ids=lambda p: str(p.relative_to(ROOT)))
def test_updated_at_is_bumped_when_the_doc_changes(path: Path) -> None:
    """`updated-at` must be no older than the last commit that touched the doc (author date survives rebases)."""
    relative = str(path.relative_to(ROOT))
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
