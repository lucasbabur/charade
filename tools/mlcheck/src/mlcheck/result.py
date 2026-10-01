"""Outcome and result types shared by every check."""

from enum import StrEnum

from pydantic import BaseModel, Field


class Status(StrEnum):
    """Final status of a check."""

    PASS = "pass"  # noqa: S105 - status label, not a secret
    FAIL = "fail"
    WARN = "warn"
    SKIP = "skip"


class Severity(StrEnum):
    """ERROR failures block the release; WARNING failures are reported as WARN."""

    ERROR = "error"
    WARNING = "warning"


class Stage(StrEnum):
    """Pipeline stage a check belongs to; also the unit of selection on the CLI."""

    STATIC = "static"
    DATA = "data"
    REPRO = "repro"
    LEAKAGE = "leakage"
    MODEL = "model"
    SERVING = "serving"
    POLICY = "policy"
    DRIFT = "drift"


class Outcome(BaseModel):
    """What a check function returns: pass or fail plus evidence."""

    ok: bool
    message: str
    details: list[str] = Field(default_factory=list)
    skipped: bool = False


class CheckResult(BaseModel):
    """Outcome after severity mapping, as reported to the user."""

    code: str
    name: str
    stage: Stage
    status: Status
    message: str
    details: list[str]


def passed(message: str, details: list[str] | None = None) -> Outcome:
    """Build a passing outcome."""
    return Outcome(ok=True, message=message, details=details or [])


def failed(message: str, details: list[str] | None = None) -> Outcome:
    """Build a failing outcome."""
    return Outcome(ok=False, message=message, details=details or [])


def skipped(message: str) -> Outcome:
    """Build an outcome for a check that does not apply to this project."""
    return Outcome(ok=True, message=message, skipped=True)
