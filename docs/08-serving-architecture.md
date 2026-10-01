# 08 — Serving architecture (< 50 ms p99)

**Bottom line:** 100 candidates ranked at **p50 8 / p99 26 ms, 400 rps, 0 errors** (Locust against `docker compose`, 8 workers, one 24-core desktop, not Fargate), with exact train/serve feature parity ([latency.json](../reports/serving/latency.json)).

## Request path

```
client ──POST /v1/rank──▶ FastAPI worker (1 of N, single-threaded libs)
                           │ 1. validate (pydantic): ≤ 500 candidates, typed ids, ISO or YYMMDDHH hour
                           │ 2. character row ← in-memory table (5k rows); unknown → request metadata or strict defaults
                           │ 3. user history ← Redis GET (20 ms budget) ──timeout──▶ cold-user defaults, degraded=true
                           │ 4. assemble N rows → charade.features.derive + encode (the training code)
                           │ 5. ONNX Runtime, one batched call → logits → isotonic calibration
                           │ 6. policy: gates → value → Beta intervals → greedy / Thompson → propensity
                           │ 7. structured decision log (request id, chosen, propensity, versions)
                           ▼
                      RankResponse (ranked list, chosen_id, propensity, confidence, cold_start, degraded)

served impression ──POST /v1/events/impression──▶ Redis user history (counters for the next request)
```

| Stage (in-process, N = 100) | p50 | p99 | How it stays small |
|---|---|---|---|
| Parse | 0.12 ms | 0.18 ms | pydantic v2 |
| Assemble (derive + encode) | 1.7 ms | 2.4 ms | Shared polars code; encode switches to dict lookups below 2,000 rows (polars `replace_strict` rebuilds hash maps per call, 2.5 ms → 0.3 ms) |
| Score | 1.6 ms | 2.0 ms | ONNX Runtime CPU, one call for all candidates, 1 intra-op thread |
| Policy | 0.8 ms | 1.3 ms | Analytic Beta intervals (normal approximation), 64 Thompson draws only for propensities |
| Redis GET | ~0.3 ms | bounded at 20 ms | One key per user; the timeout degrades, never fails |

Two measured fixes got p99 under budget:
1. **Encode:** dict lookups for small frames. The first measurement was p99 120 ms at saturation.
2. **One thread per library per worker** (`POLARS_MAX_THREADS=1` etc. in the image). Polars defaults to one thread per core, so 8 workers × 24 threads contended; p99 dropped from 48 ms to 26–30 ms at the same load.

## What is cached, precomputed, approximated

| Kind | What | Why |
|---|---|---|
| Precomputed offline | Model (ONNX), calibrator, vocabularies + scaling (`feature_spec.json`), character table, (campaign, genre) evidence | Nothing is fitted or joined at request time |
| In process memory | All of the above (~10 MB per worker) | No network hop except the user history |
| Online store | Per-user counters: totals, last two active hours, 24 h hourly buckets, per-campaign (count, last hour, count in last hour), capped at 200 campaigns | Exactly reproduces the offline counter definitions (parity tests, including hypothesis-generated event sequences) |
| Approximated | Beta posterior quantiles (normal approximation); Thompson propensity from 64 draws (≥ 1/64 resolution); exposure counts at hour granularity | Each costs microseconds instead of milliseconds; each is documented where it is used |
| Not on the hot path | Text embeddings, character enrichment, retraining, OPE | Offline jobs; serving reads their outputs |

## Failure modes

| Condition | Behaviour | Signal |
|---|---|---|
| Redis slow or down | Serve with cold-user counters, `degraded: true`; impression events → 503 (retry upstream) | `charade_degraded_total` |
| Unknown character | Use `character` metadata from the request if present; else OOV metadata and the **mature** tier (strictest brand safety) | `charade_cold_start_total{entity="character"}` |
| Unknown categorical values | OOV embedding (index 0) | Drift monitor (PSI) |
| Every candidate gated | 200, `chosen_id: null`, propensity null | `charade_no_fill_total` |
| Non-finite score | Candidate dropped with a warning; all dropped → 503 | `charade_score_errors_total` |
| Duplicate candidate ids / turn > session length | Deduplicated / clamped, with warnings in the response | — |
| No model bundle | `/health` 200, `/ready` 503: the load balancer keeps the task out of rotation | — |
| Out-of-order events | Recency is computed from strictly earlier hours only; never negative (found by the load test, fixed, regression test) | — |

## API

- `POST /v1/rank`: rank candidates for one chat moment.
- `POST /v1/events/impression`: feed back served impressions and clicks.
- `GET /v1/model`: loaded version, groups and calibrator.
- `GET /health`, `GET /ready`: liveness and readiness.
- `GET /metrics`: Prometheus.

The contract is generated from the code into [docs/api/openapi.json](api/openapi.json), and CI fails if it is stale.

## Reproduce

```bash
uv run poe train && uv run poe parity          # bundle + parity.json
docker compose up -d --build                   # API (8 workers) + Redis, bundle mounted read-only
uv run poe loadtest                            # 16 users x 25 rps, 90 s -> latency.json
USERS=32 uv run poe loadtest                   # capacity probe
```

## Scaling past one box

- **Stateless workers:** scale horizontally behind the ALB (ECS Fargate, [09-operations.md](09-operations.md)). Autoscale on CPU at about 50 % and on request count per target.
- **Redis:** ElastiCache with replicas. The per-user key design shards cleanly.
- **Larger N or models:** the model is about 2 ms of the 26 ms p99. Before reaching for GPUs, the levers are int8 ONNX quantization and candidate-count caps.
