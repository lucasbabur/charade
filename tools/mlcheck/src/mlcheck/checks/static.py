"""Static checks on source code: import boundaries and ML anti-patterns.

A finding on a line containing `mlcheck: ignore[CODE]` is suppressed, which forces a reviewed,
greppable exception (e.g. fitting the calibrator on validation data is intended).
"""

import ast
import importlib
import sys
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from functools import cache
from pathlib import Path

import grimp

from mlcheck.context import ArtifactMissingError, Context, NotConfiguredError
from mlcheck.registry import check
from mlcheck.result import Outcome, Stage, failed, passed

_UNSAFE_LOADERS = frozenset(
    {"pickle.load", "pickle.loads", "joblib.load", "dill.load", "dill.loads", "cloudpickle.load", "cloudpickle.loads"}
)
_SKIP_DIRS = frozenset({".venv", "venv", "node_modules", ".git", ".ipynb_checkpoints", "build", "dist"})


def _package(ctx: Context) -> str:
    if ctx.config.package is None:
        raise NotConfiguredError("[tool.mlcheck] package not configured")
    return ctx.config.package


def _source_files(ctx: Context) -> list[Path]:
    base = ctx.root / ctx.config.source_root / _package(ctx).replace(".", "/")
    if not (base / "__init__.py").is_file():
        raise ArtifactMissingError(f"source package {base.relative_to(ctx.root)} not found")
    return sorted(base.rglob("*.py"))


def _aliases(tree: ast.Module) -> dict[str, str]:
    """Map local names to fully qualified module paths (``np`` -> ``numpy``)."""
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                aliases[alias.asname or alias.name.split(".")[0]] = (
                    alias.name if alias.asname else alias.name.split(".")[0]
                )
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            for alias in node.names:
                aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return aliases


def _dotted(node: ast.expr) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _qualified_call(call: ast.Call, aliases: dict[str, str]) -> str | None:
    dotted = _dotted(call.func)
    if dotted is None:
        return None
    head, _, rest = dotted.partition(".")
    resolved = aliases.get(head, head)
    return f"{resolved}.{rest}" if rest else resolved


def _calls(ctx: Context) -> Iterator[tuple[Path, ast.Call, str | None, list[str]]]:
    for path in _source_files(ctx):
        source = path.read_text()
        tree = ast.parse(source, filename=str(path))
        aliases = _aliases(tree)
        lines = source.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                yield path, node, _qualified_call(node, aliases), lines


def _suppressed(lines: list[str], node: ast.AST, code: str) -> bool:
    line = lines[getattr(node, "lineno", 1) - 1]
    return f"mlcheck: ignore[{code}]" in line


def _where(ctx: Context, path: Path, node: ast.AST) -> str:
    return f"{path.relative_to(ctx.root)}:{getattr(node, 'lineno', 0)}"


type _Matcher = Callable[[ast.Call, str | None], str | None]


def _scan(ctx: Context, code: str, matcher: _Matcher) -> list[str]:
    findings: list[str] = []
    for path, call, name, lines in _calls(ctx):
        message = matcher(call, name)
        if message and not _suppressed(lines, call, code):
            findings.append(f"{_where(ctx, path, call)} {message}")
    return findings


@contextmanager
def _on_path(directory: Path) -> Generator[None]:
    sys.path.insert(0, str(directory))
    importlib.invalidate_caches()
    try:
        yield
    finally:
        sys.path.remove(str(directory))


@cache
def _graph(source_dir: Path, package: str) -> grimp.ImportGraph:
    with _on_path(source_dir):
        return grimp.build_graph(package, include_external_packages=True, cache_dir=None)


def _modules_under(graph: grimp.ImportGraph, package: str) -> set[str]:
    return {m for m in graph.modules if m == package or m.startswith(f"{package}.")}


def _serving_modules(ctx: Context) -> tuple[grimp.ImportGraph, set[str]]:
    if ctx.config.serving_package is None:
        raise NotConfiguredError("[tool.mlcheck] serving_package not configured")
    _source_files(ctx)
    graph = _graph((ctx.root / ctx.config.source_root).resolve(), _package(ctx))
    serving = _modules_under(graph, ctx.config.serving_package)
    return graph, serving


@check(
    "MLS001",
    "serving-no-training-deps",
    Stage.STATIC,
    "The serving path must not import training frameworks, even transitively: they bloat the image, slow cold "
    "starts and invite fitting code into the request path.",
)
def serving_no_training_deps(ctx: Context) -> Outcome:
    """No serving module reaches a training-only package through any import chain."""
    graph, serving = _serving_modules(ctx)
    if not serving:
        return failed(f"serving package {ctx.config.serving_package} not found")
    chains: list[str] = []
    for external in ctx.config.training_only_modules:
        if external not in graph.modules:
            continue
        for module in sorted(serving & graph.find_downstream_modules(external)):
            chain = graph.find_shortest_chain(importer=module, imported=external)
            chains.append(" -> ".join(chain or (module, external)))
    if chains:
        return failed(f"{len(chains)} serving modules import training-only packages", chains)
    return passed(f"{len(serving)} serving modules free of {', '.join(ctx.config.training_only_modules)}")


@check(
    "MLS002",
    "serving-uses-shared-features",
    Stage.STATIC,
    "Training and serving must build features with the same code; a serving path that re-implements features is "
    "the classic source of train/serve skew.",
)
def serving_uses_shared_features(ctx: Context) -> Outcome:
    """The serving package imports the shared feature package."""
    if ctx.config.features_package is None:
        raise NotConfiguredError("[tool.mlcheck] features_package not configured")
    graph, serving = _serving_modules(ctx)
    features = _modules_under(graph, ctx.config.features_package)
    if not features:
        return failed(f"features package {ctx.config.features_package} not found")
    users = serving & graph.find_downstream_modules(ctx.config.features_package, as_package=True)
    if not users:
        return failed(f"no serving module imports {ctx.config.features_package}")
    return passed(f"{len(users)} serving modules use {ctx.config.features_package}", sorted(users))


def _unsafe_load(call: ast.Call, name: str | None) -> str | None:
    if name in _UNSAFE_LOADERS:
        return f"{name}: arbitrary code execution on untrusted files; use safetensors/ONNX/JSON"
    if name == "torch.load":
        weights_only = next((kw.value for kw in call.keywords if kw.arg == "weights_only"), None)
        if not (isinstance(weights_only, ast.Constant) and weights_only.value is True):
            return "torch.load without weights_only=True"
    return None


@check(
    "MLS005",
    "safe-model-loading",
    Stage.STATIC,
    "Model artifacts travel through buckets and registries; pickle-based loading executes code from them.",
)
def safe_model_loading(ctx: Context) -> Outcome:
    """No pickle-family loaders and no unrestricted torch.load."""
    findings = _scan(ctx, "MLS005", _unsafe_load)
    return failed("unsafe deserialization found", findings) if findings else passed("no unsafe loaders")


@check(
    "MLS006",
    "no-notebooks",
    Stage.STATIC,
    "Notebooks hide execution order and state; outside explicitly allowed directories (whose notebooks must be "
    "executed in CI), every result must come from a re-runnable command.",
)
def no_notebooks(ctx: Context) -> Outcome:
    """No .ipynb files outside `notebooks_allowed_in`."""
    allowed = set(ctx.config.notebooks_allowed_in)
    found = [
        str(p.relative_to(ctx.root))
        for p in ctx.root.rglob("*.ipynb")
        if not _SKIP_DIRS.intersection(p.relative_to(ctx.root).parts)
        and p.relative_to(ctx.root).parts[0] not in allowed
    ]
    return failed(f"{len(found)} notebooks in repo", found) if found else passed("no notebooks")
