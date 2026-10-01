"""Prometheus metrics (scraped at /metrics)."""

from prometheus_client import Counter, Histogram

LATENCY = Histogram(
    "charade_request_seconds",
    "Request latency by route",
    ["route"],
    buckets=(0.001, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.1, 0.25),
)
STAGE = Histogram(
    "charade_rank_stage_seconds",
    "Latency of each /v1/rank stage",
    ["stage"],
    buckets=(0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05),
)
PCTR = Histogram("charade_pctr", "Served pCTR of the chosen candidate", buckets=tuple(i / 20 for i in range(1, 20)))
CANDIDATES = Histogram("charade_candidates", "Candidates per request", buckets=(1, 2, 5, 10, 20, 50, 100, 200, 500))
COLD = Counter("charade_cold_start_total", "Requests with an entity the system has no history for", ["entity"])
DEGRADED = Counter("charade_degraded_total", "Requests served with feature-store defaults")
GATED = Counter("charade_gated_total", "Gated candidates by reason", ["reason"])
EXPLORED = Counter("charade_explored_total", "Decisions taken by the exploration bucket")
NO_FILL = Counter("charade_no_fill_total", "Requests where every candidate was gated")
SCORE_ERRORS = Counter("charade_score_errors_total", "Candidates dropped for non-finite scores")
