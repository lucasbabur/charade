---
title: "Serving and operations"
created-at: 2026-10-01
updated-at: 2026-10-02
---

# 07 — Serving and operations (< 50 ms p99)

**Bottom line:** 100 candidates ranked at **p50 8 / p99 25 ms, 400 rps for 90 s**, with impression and click events in the mix (Locust against `docker compose`, 8 workers, one 24-core desktop, not Fargate), with exact train/serve feature parity ([latency.json](../reports/serving/latency.json)). Errors: 2 of 39,839 requests, both impression events answered 503 when Redis exceeded the 10 ms store budget (callers retry; the contract below); 7 of 35,611 rankings served `degraded` for the same reason. The report names the bundle digest the tested API served (mlcheck MLR005). **Scope:** 100 candidates, not the 500 the API accepts; impressions reported for 10 % of rankings; one desktop, not Fargate. A Fargate run with production event volume comes before promising an SLA.

## Request path

```
client ──POST /v1/rank──▶ FastAPI worker (1 of N, single-threaded libs)
                           │ 1. validate (pydantic): ≤ 500 candidates, typed ids, ISO or YYMMDDHH hour
                           │ 2. character row ← in-memory table (5k rows); unknown → request metadata or strict defaults
                           │ 3. user history ← Redis GET (20 ms budget) ──timeout──▶ cold-user defaults, degraded=true
                           │ 4. assemble N rows → charade.features.derive + encode (the training code)
                           │ 5. ONNX Runtime, one batched call → logits → calibration map (identity for the shipped model)
                           │ 6. policy: gates → value → evidence intervals → greedy / 5 % exploration → exact propensities
                           │ 7. store the decision: request_id → {user, hour, chosen ad, campaign}, write-once (Redis SET NX, 48 h)
                           │ 8. JSON decision log: context, every candidate's attributes, pCTR, gates, propensity; chosen id; versions
                           ▼
                      RankResponse (ranked list, chosen_id, propensity, confidence, cold_start, degraded)

served impression ──POST /v1/events/impression {impression_id, request_id}──▶ identity from the stored decision ──▶ user history (once per id, atomic)
late click ────────POST /v1/events/click {impression_id}──────────────────▶ attributed to the impression's hour, once (within 48 h)
```

| Stage (in-process, N = 100) | p50 | p99 | How it stays small |
|---|---|---|---|
| Parse | 0.12 ms | 0.18 ms | pydantic v2 |
| Assemble (derive + encode) | 1.7 ms | 2.4 ms | Shared polars code; encode switches to dict lookups below 2,000 rows (polars `replace_strict` rebuilds hash maps per call, 2.5 ms → 0.3 ms) |
| Score | 1.6 ms | 2.0 ms | ONNX Runtime CPU, one call for all candidates, 1 intra-op thread |
| Policy | 0.8 ms | 1.3 ms | Analytic intervals (normal approximation); closed-form exploration distribution |
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
| Approximated | Evidence-interval quantiles (normal approximation); exposure counts at hour granularity; user detail kept for 48 h (snapshots more than 24 h behind a user's newest event see pruned detail) | Each costs microseconds instead of milliseconds; each is documented where it is used |
| Not on the hot path | Text embeddings, character enrichment, retraining, OPE | Offline jobs; serving reads their outputs |

## Failure modes

| Condition | Behaviour | Signal |
|---|---|---|
| Redis slow or down | Serve with cold-user counters, `degraded: true`; **frequency cap not enforced** and the decision not stored (warning in the response), so that request's impression event later gets 404; events → 503, retried upstream (idempotent) | `charade_degraded_total`, `charade_frequency_cap_unenforced_total` |
| Unknown character | Use `character` metadata from the request if present; else OOV metadata and the **mature** tier (strictest brand safety) | `charade_cold_start_total{entity="character"}` |
| Unknown categorical values | OOV embedding (index 0) | Drift monitor (PSI) |
| Every candidate gated | 200, `chosen_id: null`, propensity null | `charade_no_fill_total` |
| Non-finite score | Candidate dropped with a warning; all dropped → 503 | `charade_score_errors_total` |
| Duplicate candidate ids / turn > session length | Deduplicated / clamped, with warnings in the response | — |
| No model bundle | `/health` 200, `/ready` 503: the load balancer keeps the task out of rotation | — |
| Out-of-order events | Every counter uses strictly earlier hours, whatever the arrival order (property test over random arrival orders) | — |
| Retried or duplicated events | No-op: decisions are keyed by `request_id`, impressions and clicks by `impression_id` | — |
| Conflicting retries | 409: a `request_id` re-ranked into a different decision, or an `impression_id` reused for another request. A retry can never change what an event means | — |
| Impression for an unknown request | 404: user, campaign and hour come only from the stored decision, so an event cannot credit an impression to the wrong user or campaign | — |
| Concurrent writes for one user | Redis WATCH/MULTI transactions with retry; no lost updates (concurrency test on fakeredis) | — |

## API

- `POST /v1/rank`: rank candidates for one chat moment.
- `POST /v1/events/impression {impression_id, request_id}` and `POST /v1/events/click {impression_id}`: feed back served impressions and clicks, idempotent on `impression_id` (duplicates return `{"outcome": "duplicate"}`; unknown requests or clicks 404; conflicting reuse 409). The impression takes its user, campaign and hour from the decision, never from the caller.

**One identity per event, online and offline.** The online history and the training rows rebuilt from logs (`charade.data.events`) use the same identities: the impression's ad is its decision's chosen candidate. The row builder keeps the first copy of each identical redelivery and drops, by reason, impressions with no decision, conflicting decisions or impressions, and impressions whose ad is not the one served.

**Label maturity.** A click is attributed only within 48 h of its impression (the decision and impression keys expire then), so an impression becomes a training row only after that window closes; younger ones are counted as `label pending`, not written as `click = 0`. The model's user counters still assume clicks from earlier hours were known at serve time (`charade.features.counters`): offline they include a 09:55 impression's 10:10 click in a 10:00 request, online that click arrives later. Measuring the cost needs real click delays, which the data does not have.
- `GET /v1/model`: loaded version, groups and calibrator.
- `GET /health`, `GET /ready`: liveness and readiness.
- `GET /metrics`: Prometheus.

The contract is generated from the code into [docs/api/openapi.json](api/openapi.json), and CI fails if it is stale.

## Reproduce

```bash
uv run poe train && uv run poe parity          # bundle + parity.json
docker compose up -d --build                   # API (8 workers) + Redis, bundle mounted read-only
USERS=16 uv run poe loadtest                   # 16 users x 25 rps, 90 s -> latency.json (default is 32 users)
USERS=32 uv run poe loadtest                   # capacity probe
```

## Scaling past one box

- **Stateless workers:** scale horizontally behind the ALB (ECS Fargate, [07-serving-operations.md](07-serving-operations.md)). Autoscale on CPU at about 50 % and on request count per target.
- **Redis:** ElastiCache with replicas. The per-user key design shards cleanly.
- **Larger N or models:** the model is about 2 ms of the 25 ms p99. Before reaching for GPUs, the levers are int8 ONNX quantization and candidate-count caps.
- **Decision logs at Simula's volume (~1.2 B ads served):** logging every candidate of every request (≈ 10 KB for N = 100) is ≈ 12 TB of mostly redundant data. Log the served ad, its propensity and the context on every request (a few hundred bytes); log full candidate sets only for the exploration bucket plus a small random sample of greedy requests, which is all off-policy evaluation and counterfactual training need. Write asynchronously (stdout → log agent → object storage, never in the request path) in a compact columnar format (Parquet or protobuf), not JSON.

## Infrastructure (Terraform sketch)

 a Terraform **sketch** of the serving environment (VPC, ECS API behind an internal ALB, Redis, alarms), validated with `terraform validate`, tflint and checkov, **never applied**. Retraining, promotion and the decision-log export are scripts, not managed jobs. A first deployment would surface what static checks cannot (networking, IAM); this page does not claim a working platform.

### Topology

```
ad servers ──HTTPS──▶ internal ALB (TLS 1.3, /ready health) ──▶ ECS Fargate API tasks (8 vCPU, 8 workers, 3–30 tasks, 3 AZs)
                                                                  │  init container: copy bundles/<run_id>/ (run id pinned in the task definition) → task volume (read-only)
                                                                  ├──▶ ElastiCache Redis 7 (multi-AZ, TLS + AUTH, user history)
                                                                  └──▶ stdout JSON decision / impression / click events ─▶ CloudWatch Logs
scripts/retrain.sh (run by hand or any scheduler): pull exports → rolling windows → train → ope → parity → drift → mlcheck → scripts/promote.sh
```

| Module | What it owns | Notable choices |
|---|---|---|
| `platform` | Composition, KMS CMK (rotation on), VPC via `terraform-aws-modules/vpc` (3 pinned AZ ids, NAT, flow logs) | AZs pinned by zone id so subnets never shift |
| `bucket` | Private, versioned, encrypted buckets (artifacts, logs) | TLS-only policy, owner-enforced ACLs, lifecycle; SSE-S3 only for ALB logs (an AWS constraint) |
| `ecr` | `api` repository | Immutable tags, scan on push, KMS, keep 20 |
| `ecs_api` | Cluster, ALB, target group, service, autoscaling, IAM | Bundle run id pinned in the task definition, so the circuit breaker's rollback also restores the previous model; scales on CPU 50 % and on 400 rps per task (the load-tested point); read-only root filesystem; Redis URL from Secrets Manager |
| `redis` | Replication group, subnet group, security group, URL secret | Reachable only from API tasks; encrypted at rest and in transit |
| `observability` | SNS topic, alarms, dashboard | Alarms come from ALB metrics and log metric filters, so no metrics agent is needed |

**Deliberately not built:** a scheduled training job, a log-export stream and CI deploy credentials to AWS. Earlier versions had all three; static review found real defects in them (security groups blocking AWS endpoints, OIDC trust not matching environment jobs, a deploy role without Terraform permissions) that only a real deployment would settle, so they were removed rather than shipped as if they worked.

## Delivery

| What | Trigger | Does |
|---|---|---|
| `ci.yml` | Every PR and push to `main` | ruff, pyright strict, pytest (coverage ≥ 85 %), mlcheck tests and static gates, OpenAPI drift; terraform fmt/validate, tflint, checkov; actionlint; hadolint (both images), shellcheck, API image build + `/health` smoke |
| `release.yml` | Tag `v*` | Builds the API image from the tagged commit and pushes it to GHCR as `charade-api:<version>` and `:sha-<commit>`. Delivery stops at this versioned artifact; nothing is deployed |
| `scripts/retrain.sh` | By hand (daily in production) | Split windows derived from the export (`charade.data.windows`), train, evaluate, mlcheck, then `scripts/promote.sh`. Verified locally: the training image trains on CPU and passes the run gates; the AWS steps have never run |
| `scripts/promote.sh <run_id>` | After passing gates, or for rollback | Registers an API task definition revision pinned to that bundle, deploys it, waits for the rollout; a failed rollout is rolled back, with its model, by the circuit breaker |
| `python -m charade.data.events` | Before retraining | Turns exported decision/impression/click logs (JSON lines, gzipped or not) into `impressions.csv` rows: deduplicated, conflicts dropped by reason, labels younger than the 48 h click window held back |
| Dependabot | Weekly | Actions, uv, Docker base images (Python minor/major pinned), Terraform providers |

**Model promotion is separate from code deploys.** A code release changes how bundles are served. Retraining changes which bundle is served, and only through the mlcheck gates (leakage, calibration, beats-baseline CI, parity, policy invariants). A failed gate stops the script before promotion, and production keeps yesterday's bundle.

## Alarms (`modules/observability`)

| Alarm | Condition | First move |
|---|---|---|
| `api-p99-latency` | ALB p99 > 40 ms for 5 min (SLO 50 ms) | See [Latency](#latency) |
| `api-5xx-rate` | Target 5xx > 0.5 % for 5 min | See [Errors](#errors) |
| `degraded` | > 50 decisions in 5 min served without the feature store | Redis health: failover, connections, CPU; serving continues with cold-user defaults |
| `no_fill` | > 500 all-gated decisions in 5 min | A brand-safety matrix change or exhausted budgets; check `gate_reasons` in the decision logs |

**What else to watch** (Prometheus at `/metrics`, or the decision logs):
- pCTR distribution against the training snapshot
- calibration ratio once clicks join, alarming outside [0.9, 1.1]
- cold-start, exploration and gate rates by reason
- per-stage latency histograms (`charade_rank_stage_seconds`)

## Runbooks

### Latency
1. Read `charade_rank_stage_seconds` by stage: `fetch` (Redis), `assemble`, `score`, `policy`.
2. If `fetch` is high, check the Redis CPU and connection count, or a failover in progress. Raising `store_timeout_ms` trades latency for fewer degraded responses.
3. If CPU is high and the stages are flat, it is queueing: check the autoscaling activity and `max_tasks`. Measured capacity is ~750 rps per task at p99 75 ms.
4. If a new release coincides, roll back (the circuit breaker does this for failed health checks).

### Errors
- **5xx with `model bundle not loaded`:** the init container could not copy the bundle the revision pins (`BUNDLE_RUN_ID`). Check S3 permissions and that `bundles/<run_id>/` exists.
- **`no candidate could be scored`:** non-finite inputs. Inspect the request (the fixture test shows the guard). The API never returns 5xx for Redis outages.

### Retraining
1. Read the run's `mlcheck` output. Each failing code links to [mlcheck.md](mlcheck.md).
2. Data gates (MLD*): an upstream export changed. Fix the export; do not relax the contract.
3. Model gates (MLM*): compare `runs/<run_id>/` with the previous run's `metrics.json`. A real regression needs investigation, not a forced promotion.
4. To promote or roll back manually: `scripts/promote.sh <run_id>`. It registers a revision pinned to that bundle, deploys and waits; rolling back the task definition alone also rolls back the model.

### Redis
Losing Redis loses user history for a few hours of degraded serving; it is not an outage. Rebuild from the last 24 h of decision logs and impression events if needed.

## Cost sketch (us-east-1, steady 400 rps)

| Item | Approximate monthly cost |
|---|---|
| 3 × API tasks (8 vCPU / 16 GB, Fargate) | ~$1,050 |
| Redis `cache.r7g.large` × 3 | ~$480 |
| NAT gateways (3) | ~$100 |
| ALB | ~$25 |
| Daily training (16 vCPU, about 30 min) | < $10 |
| S3 and CloudWatch Logs | Tens of dollars |

These are order-of-magnitude figures from public on-demand prices; verify before budgeting. Savings Plans and Graviton (`cpu_architecture = "ARM64"`, onnxruntime supports it) are the first levers.
