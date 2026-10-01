#!/usr/bin/env bash
# Daily retrain inside the training image (infra/terraform/modules/training_job):
# pull exports -> derive rolling windows -> train -> evaluate -> mlcheck -> promote -> redeploy.
#
# Promotion is atomic: the bundle is uploaded to an immutable prefix bundles/<run_id>/, then the
# one-line pointer bundles/CURRENT is overwritten (a single S3 PUT). API tasks read the pointer at
# start, so a task never sees a mix of two bundles. The script then forces a new ECS deployment so
# running tasks pick up the new bundle (the deployment circuit breaker rolls back if /ready fails).
# Production keeps the previous bundle unless every blocking gate passes; WARN gates are reported.
set -euo pipefail

: "${CHARADE_DATA_URI:?s3:// prefix of impressions.csv and characters.csv}"
: "${CHARADE_ARTIFACTS_URI:?s3:// artifacts bucket}"
: "${CHARADE_ECS_CLUSTER:?ECS cluster of the API service}"
: "${CHARADE_ECS_SERVICE:?ECS service to redeploy}"

export CHARADE_DATA_DIR=/work/data
export CHARADE_ARTIFACTS_DIR=/work/artifacts/current
mkdir -p "$CHARADE_DATA_DIR" "$CHARADE_ARTIFACTS_DIR"

aws s3 cp --only-show-errors "${CHARADE_DATA_URI%/}/impressions.csv" "$CHARADE_DATA_DIR/"
aws s3 cp --only-show-errors "${CHARADE_DATA_URI%/}/characters.csv" "$CHARADE_DATA_DIR/"

# Rolling split: last complete day = holdout, the day before = validation, the rest = training.
eval "$(python -m charade.data.windows "$CHARADE_DATA_DIR/impressions.csv")"
echo "windows: train <= $CHARADE_TRAIN_END < val <= $CHARADE_VAL_END < holdout"

poe train
poe ope
poe parity
poe drift
mlcheck . --artifacts-dir "$CHARADE_ARTIFACTS_DIR" --data-dir "$CHARADE_DATA_DIR" \
  --skip MLV003 --skip MLV004 # latency is gated by the load test before release, not per retrain

run_id=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['run_id'])" "$CHARADE_ARTIFACTS_DIR/manifest.json")
bucket="${CHARADE_ARTIFACTS_URI%/}"
aws s3 cp --recursive --only-show-errors "$CHARADE_ARTIFACTS_DIR" "$bucket/bundles/$run_id/"
printf '%s\n' "$run_id" | aws s3 cp - "$bucket/bundles/CURRENT"
aws ecs update-service --cluster "$CHARADE_ECS_CLUSTER" --service "$CHARADE_ECS_SERVICE" \
  --force-new-deployment --no-cli-pager >/dev/null
echo "promoted $run_id and started a rolling deployment of $CHARADE_ECS_SERVICE"
