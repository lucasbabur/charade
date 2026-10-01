---
title: "Operations: infrastructure, delivery, monitoring, runbooks"
created-at: 2026-10-01
updated-at: 2026-10-01
---

# 09 — Operations: infrastructure, delivery, monitoring, runbooks

**Bottom line:** one Terraform module builds an AWS environment (VPC, ECS API behind an internal ALB, Redis, Firehose decision logs, daily gated retraining, alarms, GitHub OIDC); validated with tflint and checkov, never applied.

## Topology

```
ad servers ──HTTPS──▶ internal ALB (TLS 1.3, /ready health) ──▶ ECS Fargate API tasks (8 vCPU, 8 workers, 3–30 tasks, 3 AZs)
                                                                  │  init container: read bundles/CURRENT → copy bundles/<run_id>/ → task volume (read-only)
                                                                  ├──▶ ElastiCache Redis 7 (multi-AZ, TLS + AUTH, user history)
                                                                  └──▶ stdout JSON "decision" events ─▶ CloudWatch Logs ─▶ Firehose ─▶ S3 data/decisions/dt=/hour=
EventBridge Scheduler 02:30 UTC ──▶ Fargate training task (16 vCPU): pull exports → rolling windows → train → ope → parity → drift → mlcheck → promote → redeploy
GitHub Actions ──OIDC──▶ deploy role (ECR push, ECS update, PassRole on task roles only)
```

| Module | What it owns | Notable choices |
|---|---|---|
| `platform` | Composition, KMS CMK (rotation on), VPC via `terraform-aws-modules/vpc` (3 pinned AZ ids, NAT, flow logs) | AZs pinned by zone id so subnets never shift |
| `bucket` | Private, versioned, encrypted buckets (artifacts, data, logs) | TLS-only policy, owner-enforced ACLs, lifecycle; SSE-S3 only for ALB logs (an AWS constraint) |
| `ecr` | `api` and `train` repositories | Immutable tags, scan on push, KMS, keep 20 |
| `ecs_api` | Cluster, ALB, target group, service, autoscaling, IAM | Circuit-breaker rollback; scales on CPU 50 % and on 400 rps per task (the load-tested point); read-only root filesystem; Redis URL injected from Secrets Manager |
| `redis` | Replication group, subnet group, security group, URL secret | Reachable only from API tasks; encrypted at rest and in transit |
| `decision_logs` | Firehose stream, subscription filter on the `decision`, `impression` and `click` events | GZIP, hourly partitions, KMS; `charade.data.events` joins them into `impressions.csv`-shaped training rows: Firehose decompresses the CloudWatch subscription envelopes and extracts each message, so S3 holds gzipped JSON lines; the builder reads them, deduplicates and holds back labels younger than the 48 h click window (tested end to end against the API; the scheduled job that runs it on S3 is not built) |
| `training_job` | Task definition, scheduler, roles | Uploads an immutable `bundles/<run_id>/`, then flips the one-line `bundles/CURRENT` pointer and forces a rolling redeploy, only after mlcheck passes |
| `observability` | SNS topic, alarms, dashboard | Alarms come from ALB metrics and log metric filters, so no metrics agent is needed |
| `ci_oidc` | GitHub OIDC provider and deploy role | Trust limited to `main` and `v*` tags of this repository |

## Delivery

| Workflow | Trigger | Does |
|---|---|---|
| `ci.yml` | Every PR and push to `main` | ruff, pyright strict, pytest (coverage ≥ 85 %), mlcheck tests and static gates, OpenAPI drift; terraform fmt/validate, tflint, checkov; actionlint; hadolint (both images), shellcheck, API image build + `/health` smoke |
| `cd.yml` | Tag `v*` or manual | Build and push both images (commit baked in), then `terraform plan` + `apply` for staging and then prod behind GitHub environment approvals. Inert until the repository variable `AWS_DEPLOY_ROLE_ARN` exists |
| Daily retrain | 02:30 UTC | `scripts/retrain.sh` in the training image: split windows derived from the export (`charade.data.windows`), gated promotion, redeploy. Verified locally: the image trains on CPU and passes the run gates; the AWS steps have never run |
| Dependabot | Weekly | Actions, uv, Docker base images (Python minor/major pinned), Terraform providers |

**Model promotion is separate from code deploys.** A code release changes how bundles are served. Retraining changes which bundle is served, and only through the mlcheck gates (leakage, calibration, beats-baseline CI, parity, policy invariants). A failed gate pages, and production keeps yesterday's bundle.

## Alarms (`modules/observability`)

| Alarm | Condition | First move |
|---|---|---|
| `api-p99-latency` | ALB p99 > 40 ms for 5 min (SLO 50 ms) | See [Latency](#latency) |
| `api-5xx-rate` | Target 5xx > 0.5 % for 5 min | See [Errors](#errors) |
| `degraded` | > 50 decisions in 5 min served without the feature store | Redis health: failover, connections, CPU; serving continues with cold-user defaults |
| `no_fill` | > 500 all-gated decisions in 5 min | A brand-safety matrix change or exhausted budgets; check `gate_reasons` in the decision logs |
| `retrain-gate-failure` | The daily run reported a failing mlcheck gate | See [Retraining](#retraining) |

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
- **5xx with `model bundle not loaded`:** the init container could not resolve `bundles/CURRENT` or copy the bundle it names. Check S3 permissions and that the pointer names an existing prefix.
- **`no candidate could be scored`:** non-finite inputs. Inspect the request (the fixture test shows the guard). The API never returns 5xx for Redis outages.

### Retraining
1. Open the run's `mlcheck` output in the training log group. Each failing code links to [mlcheck.md](mlcheck.md).
2. Data gates (MLD*): an upstream export changed. Fix the export; do not relax the contract.
3. Model gates (MLM*): compare `runs/<run_id>/` with the previous run's `metrics.json`. A real regression needs investigation, not a forced promotion.
4. To promote or roll back manually: `echo <run_id> | aws s3 cp - s3://<artifacts>/bundles/CURRENT`, then force a new ECS deployment.

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
| S3 and Firehose | Tens of dollars |

These are order-of-magnitude figures from public on-demand prices; verify before budgeting. Savings Plans and Graviton (`cpu_architecture = "ARM64"`, onnxruntime supports it) are the first levers.
