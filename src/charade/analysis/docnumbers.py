"""Doc numbers generated from reports (`uv run poe docs-numbers`; `--check` fails on drift).

A number in the docs written as `<!--n:key-->value<!--/n-->` is filled from `reports/` by this
module, so prose stays hand-written but the figures cannot drift from the evidence. `tests/docs`
runs the check in CI, the same pattern as the generated OpenAPI contract.
"""

import json
import re
import sys
from collections.abc import Callable
from pathlib import Path

MINUS = "\u2212"
"""Typographic minus, as the docs print negative numbers."""
MARKER = re.compile(r"<!--n:([a-z0-9_]+)-->(.*?)<!--/n-->")
DOCS = ("docs/*.md", "experiments/*/README.md")


def _signed(value: float, digits: int = 4) -> str:
    return f"{value:+.{digits}f}".replace("-", MINUS)


def _interval(delta: float, low: float, high: float, digits: int = 4) -> str:
    return f"{_signed(delta, digits)} [{_signed(low, digits)}, {_signed(high, digits)}]"


def _ope_dr_lift(text: str, policy: str) -> tuple[float, float, float]:
    """DR lift (fraction) and CI for `policy` from the lift table of `reports/policy/ope.md`."""
    table = text[text.index("## Lift over the logging policy") :]
    row = next(line for line in table.splitlines() if line.startswith(f"| {policy} |"))
    dr = row.split("|")[3].strip()
    point, low, high = (float(x) for x in re.findall(r"[+-]\d+\.\d+", dr))
    return point, low, high


def numbers(root: Path) -> dict[str, str]:
    """Every generated doc number, formatted as the docs print it."""
    reports = root / "reports"
    metrics = json.loads((reports / "models/metrics.json").read_text())
    test = metrics["models"]["dcn_v2"]["test"]
    comparisons = metrics["comparisons_test"]
    slices = {(r["slice"], r["value"]): r for r in metrics["slices_test"]}
    flat = json.loads((reports / "models/opportunity.json").read_text())["logloss_cost_of_flattening"]
    ope = (reports / "policy/ope.md").read_text()
    lift = _ope_dr_lift(ope, "shipped policy (gates + 5% exploration)")
    cold, warm = slices[("character_support", "0 (cold)")], slices[("character_support", ">200")]
    new, seen = slices[("user_seen", "new")], slices[("user_seen", "seen")]

    def cmp(name: str) -> str:
        c = comparisons[name]
        return _interval(c["delta_logloss"], c["ci_low"], c["ci_high"])

    values: dict[str, Callable[[], str]] = {
        "dcn_ne": lambda: f"{test['ne']:.4f}",
        "dcn_auc": lambda: f"{test['auc']:.3f}",
        "dcn_pred_obs": lambda: f"{test['calibration_ratio']:.3f}",
        "dcn_ece": lambda: f"{test['ece']:.3f}",
        "vs_lightgbm": lambda: cmp("dcn_v2-lightgbm (3-seed ensembles, calibrated)"),
        "vs_logistic": lambda: cmp("dcn_v2-logreg"),
        "vs_lightgbm_single": lambda: cmp("dcn_v2-lightgbm (single seed each, uncalibrated)"),
        "flatten_cost": lambda: f"{flat['delta']:.4f} [{flat['ci_low']:.4f}, {flat['ci_high']:.4f}]",
        "ope_shipped_dr": lambda: (
            f"{_signed(lift[0] * 100, 2)} pp [{_signed(lift[1] * 100, 2)}, {_signed(lift[2] * 100, 2)}]"
        ),
        "cold_char_ne": lambda: f"{cold['ne']:.3f}",
        "warm_char_ne": lambda: f"{warm['ne']:.3f}",
        "cold_char_n": lambda: f"{int(cold['n']):,}",
        "new_user_ne": lambda: f"{new['ne']:.3f}",
        "seen_user_ne": lambda: f"{seen['ne']:.3f}",
    }
    return {key: make() for key, make in values.items()}


def _files(root: Path) -> list[Path]:
    return sorted(p for pattern in DOCS for p in root.glob(pattern))


def render(text: str, values: dict[str, str]) -> str:
    """Fill every marker; unknown keys are an error, not a silent pass."""

    def fill(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values:
            raise KeyError(f"unknown doc number {key!r}")
        return f"<!--n:{key}-->{values[key]}<!--/n-->"

    return MARKER.sub(fill, text)


def stale(root: Path) -> list[str]:
    """Docs whose generated numbers differ from the reports."""
    values = numbers(root)
    return [str(p.relative_to(root)) for p in _files(root) if render(p.read_text(), values) != p.read_text()]


def write(root: Path) -> list[str]:
    """Rewrite every doc with fresh numbers; return the changed files."""
    values, changed = numbers(root), []
    for path in _files(root):
        text = path.read_text()
        fresh = render(text, values)
        if fresh != text:
            path.write_text(fresh)
            changed.append(str(path.relative_to(root)))
    return changed


if __name__ == "__main__":
    here = Path.cwd()
    if sys.argv[1:2] == ["--check"]:
        drift = stale(here)
        print("\n".join(drift) or "doc numbers match reports")
        sys.exit(1 if drift else 0)
    print("\n".join(write(here)) or "no changes")
