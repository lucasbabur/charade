#!/usr/bin/env bash
# Serve bundle <run_id>: register an API task definition revision pinned to it, deploy, verify.
# Used by retrain.sh after the gates pass, and for rollback: `promote.sh <older run_id>`.
#
# The bundle's run id lives in the task definition (BUNDLE_RUN_ID of the fetch-bundle container), so a
# revision names exactly one model. If the rollout fails, the ECS circuit breaker rolls the service back
# to the previous revision, and with it to the previous model. bundles/CURRENT is only a record of the
# last successful promotion (CD reads it so a Terraform apply keeps serving the same bundle); tasks never
# read it.
set -euo pipefail

run_id="${1:?usage: promote.sh <run_id>}"
: "${CHARADE_ARTIFACTS_URI:?s3:// artifacts bucket}"
: "${CHARADE_ECS_CLUSTER:?ECS cluster of the API service}"
: "${CHARADE_ECS_SERVICE:?ECS service to redeploy}"
bundles="${CHARADE_ARTIFACTS_URI%/}/bundles"
aws s3 ls "$bundles/$run_id/manifest.json" >/dev/null || { echo "no bundle $run_id" >&2; exit 2; }

current=$(aws ecs describe-services --cluster "$CHARADE_ECS_CLUSTER" --services "$CHARADE_ECS_SERVICE" \
  --query 'services[0].taskDefinition' --output text)
pinned=$(aws ecs describe-task-definition --task-definition "$current" --query taskDefinition --output json |
  RUN_ID="$run_id" python3 -c '
import json, os, sys
td = json.load(sys.stdin)
for container in td["containerDefinitions"]:
    if container["name"] == "fetch-bundle":
        container["environment"] = [e for e in container.get("environment", []) if e["name"] != "BUNDLE_RUN_ID"]
        container["environment"].append({"name": "BUNDLE_RUN_ID", "value": os.environ["RUN_ID"]})
keep = ("family", "taskRoleArn", "executionRoleArn", "networkMode", "containerDefinitions", "volumes",
        "requiresCompatibilities", "cpu", "memory", "runtimePlatform")
print(json.dumps({k: td[k] for k in keep if k in td}))')
revision=$(aws ecs register-task-definition --cli-input-json "$pinned" \
  --query taskDefinition.taskDefinitionArn --output text)

deployment=$(aws ecs update-service --cluster "$CHARADE_ECS_CLUSTER" --service "$CHARADE_ECS_SERVICE" \
  --task-definition "$revision" --no-cli-pager \
  --query "service.deployments[?status=='PRIMARY'].id | [0]" --output text)
aws ecs wait services-stable --cluster "$CHARADE_ECS_CLUSTER" --services "$CHARADE_ECS_SERVICE" || true
state=$(aws ecs describe-services --cluster "$CHARADE_ECS_CLUSTER" --services "$CHARADE_ECS_SERVICE" \
  --query "services[0].deployments[?id=='$deployment'].rolloutState | [0]" --output text)

if [ "$state" != "COMPLETED" ]; then
  echo "rollout of $run_id ($revision) ended $state; the circuit breaker keeps $current and its bundle" >&2
  exit 1
fi
printf '%s\n' "$run_id" | aws s3 cp - "$bundles/CURRENT"
echo "serving $run_id via $revision (previous revision $current)"
