---
title: "Operations: infrastructure, delivery, monitoring, runbooks"
created-at: 2026-10-01
updated-at: 2026-10-02
---

# 09 — Operations: infrastructure, delivery, monitoring, runbooks

**Bottom line:** a Terraform **sketch** of the serving environment (VPC, ECS API behind an internal ALB, Redis, alarms), validated with `terraform validate`, tflint and checkov, **never applied**. Retraining, promotion and the decision-log export are scripts, not managed jobs. A first deployment would surface what static checks cannot (networking, IAM); this page does not claim a working platform.

## Topology

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
