"""Check registry and runner."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from pydantic import ValidationError

from mlcheck.context import ArtifactMissingError, Context, NotConfiguredError
from mlcheck.result import CheckResult, Outcome, Severity, Stage, Status

type CheckFn = Callable[[Context], Outcome]


@dataclass(frozen=True)
class Check:
    """A registered check."""

    code: str
    name: str
    stage: Stage
    severity: Severity
    rationale: str
    run: CheckFn


CHECKS: dict[str, Check] = {}


def check(
    code: str, name: str, stage: Stage, rationale: str, severity: Severity = Severity.ERROR
) -> Callable[[CheckFn], CheckFn]:
    """Register a check function under a stable code."""

    def register(fn: CheckFn) -> CheckFn:
        if code in CHECKS:
            raise ValueError(f"duplicate check code {code}")
        CHECKS[code] = Check(code, name, stage, severity, rationale, fn)
        return fn

    return register


def _status(chk: Check, outcome: Outcome) -> Status:
    if outcome.skipped:
        return Status.SKIP
    if outcome.ok:
        return Status.PASS
    return Status.FAIL if chk.severity is Severity.ERROR else Status.WARN


def run_check(chk: Check, ctx: Context) -> CheckResult:
    """Run one check; missing inputs and crashes become failures, never silent passes."""
    try:
        outcome = chk.run(ctx)
    except NotConfiguredError as exc:
        outcome = Outcome(ok=True, skipped=True, message=str(exc))
    except ArtifactMissingError as exc:
        outcome = Outcome(ok=False, message=str(exc))
    except ValidationError as exc:
        outcome = Outcome(ok=False, message="artifact violates contract", details=[str(exc)])
    except Exception as exc:  # noqa: BLE001 - a crashing gate must fail loudly, not abort the run
        outcome = Outcome(ok=False, message=f"check crashed: {type(exc).__name__}: {exc}")
    return CheckResult(
        code=chk.code,
        name=chk.name,
        stage=chk.stage,
        status=_status(chk, outcome),
        message=outcome.message,
        details=outcome.details,
    )


def select(stages: Iterable[Stage], only: Iterable[str], skip: Iterable[str]) -> list[Check]:
    """Checks matching the CLI selection, in code order."""
    stage_set, only_set, skip_set = set(stages), set(only), set(skip)
    return [
        chk
        for code, chk in sorted(CHECKS.items())
        if (not stage_set or chk.stage in stage_set) and (not only_set or code in only_set) and code not in skip_set
    ]
