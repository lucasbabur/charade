"""Structured JSON logs: one object per line, `event` names the record (e.g. "decision").

CloudWatch metric filters and the Firehose subscription match on `$.event`, so the shape is part of
the operational contract (infra/terraform/modules/observability, decision_logs).
"""

import logging

import structlog


def configure_logging(level: int = logging.INFO) -> None:
    """Render structlog events as JSON with an ISO timestamp and level."""
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )
