"""Command line entry point."""

import json
from pathlib import Path
from typing import Annotated

import typer

import mlcheck.checks  # noqa: F401  # pyright: ignore[reportUnusedImport] - registers checks
from mlcheck.config import load_config
from mlcheck.context import Context
from mlcheck.registry import CHECKS, run_check, select
from mlcheck.result import CheckResult, Stage, Status

app = typer.Typer(add_completion=False, help="Release gates for ML systems.")

_MARK = {Status.PASS: "PASS", Status.FAIL: "FAIL", Status.WARN: "WARN", Status.SKIP: "SKIP"}


def _render_text(results: list[CheckResult], verbose: bool) -> str:
    lines: list[str] = []
    stage: Stage | None = None
    for result in results:
        if result.stage != stage:
            stage = result.stage
            lines.append(f"\n[{stage.value}]")
        lines.append(f"  {_MARK[result.status]}  {result.code} {result.name}: {result.message}")
        show = result.details if verbose or result.status in {Status.FAIL, Status.WARN} else []
        lines.extend(f"          {detail}" for detail in show[:25])
    counts = {s: sum(r.status is s for r in results) for s in Status}
    lines.append(
        f"\n{len(results)} checks: {counts[Status.PASS]} pass, {counts[Status.FAIL]} fail, "
        f"{counts[Status.WARN]} warn, {counts[Status.SKIP]} skip"
    )
    return "\n".join(lines)


@app.command()
def main(  # noqa: PLR0917 - typer maps parameters to CLI options
    root: Annotated[Path, typer.Argument(help="Project directory.")] = Path(),
    stage: Annotated[list[Stage] | None, typer.Option(help="Stages to run (repeatable). Default: all.")] = None,
    only: Annotated[list[str] | None, typer.Option(help="Run only these check codes.")] = None,
    skip: Annotated[list[str] | None, typer.Option(help="Skip these check codes.")] = None,
    output: Annotated[str, typer.Option("--format", help="text or json.")] = "text",
    strict: Annotated[bool, typer.Option(help="Treat warnings as failures.")] = False,
    verbose: Annotated[bool, typer.Option("-v", help="Show details for passing checks.")] = False,
    list_checks: Annotated[bool, typer.Option("--list", help="List checks and exit.")] = False,
) -> None:
    """Run ML release gates against a project and its latest run artifacts."""
    if list_checks:
        for code, chk in sorted(CHECKS.items()):
            typer.echo(f"{code}  {chk.stage.value:<8} {chk.severity.value:<7} {chk.name}: {chk.rationale}")
        raise typer.Exit
    root = root.resolve()
    ctx = Context(root, load_config(root))
    results = [run_check(chk, ctx) for chk in select(stage or [], only or [], skip or [])]
    if output == "json":
        typer.echo(json.dumps([r.model_dump(mode="json") for r in results], indent=2))
    else:
        typer.echo(_render_text(results, verbose))
    blocking = {Status.FAIL, Status.WARN} if strict else {Status.FAIL}
    raise typer.Exit(1 if any(r.status in blocking for r in results) else 0)
