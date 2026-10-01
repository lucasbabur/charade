import json
from pathlib import Path

from typer.testing import CliRunner

from mlcheck.cli import app
from mlcheck.registry import CHECKS

runner = CliRunner()


def test_golden_project_exits_zero(golden: Path) -> None:
    result = runner.invoke(app, [str(golden)])
    assert result.exit_code == 0, result.output
    assert f"{len(CHECKS)} checks: {len(CHECKS)} pass" in result.output


def test_json_output_and_stage_selection(golden: Path) -> None:
    result = runner.invoke(app, [str(golden), "--stage", "serving", "--format", "json"])
    payload = json.loads(result.output)
    assert {r["stage"] for r in payload} == {"serving"}


def test_failure_exits_nonzero(project: Path) -> None:
    (project / "artifacts/current/parity.json").unlink()
    result = runner.invoke(app, [str(project), "--only", "MLV001"])
    assert result.exit_code == 1


def test_warning_blocks_only_in_strict_mode(project: Path) -> None:
    path = project / "artifacts/current/drift.json"
    path.write_text(json.dumps({"reference": "train", "psi": {"site": {"d": 0.9}}}))
    assert runner.invoke(app, [str(project), "--only", "MLX001"]).exit_code == 0
    assert runner.invoke(app, [str(project), "--only", "MLX001", "--strict"]).exit_code == 1


def test_list_shows_every_check() -> None:
    result = runner.invoke(app, ["--list"])
    assert all(code in result.output for code in CHECKS)


def test_docs_catalogue_lists_every_check() -> None:
    catalogue = (Path(__file__).parents[3] / "docs" / "mlcheck.md").read_text()
    assert all(f"| {code} |" in catalogue for code in CHECKS)
