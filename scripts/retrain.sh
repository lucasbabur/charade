#!/usr/bin/env bash
# Daily retrain inside the training image (docker/train.Dockerfile), run by hand or by any scheduler:
# pull exports -> derive rolling windows -> train -> evaluate -> mlcheck -> promote (pinned revision).
#
# Promotion: the bundle is uploaded to an immutable prefix bundles/<run_id>/, then scripts/promote.sh
# deploys an API task definition revision pinned to that run id and waits for the rollout; a failed
# rollout is rolled back, with its bundle, by the ECS circuit breaker. Production keeps the previous
# revision and bundle unless every blocking gate passes; WARN gates are reported.
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
echo "windows: train <= $CHARADE_TRAIN_END < val <= $CHARADE_VAL_END < holdout <= $CHARADE_TEST_END"

poe train
poe ope
poe parity
poe correction
poe drift
mlcheck . --artifacts-dir "$CHARADE_ARTIFACTS_DIR" --data-dir "$CHARADE_DATA_DIR" \
  --skip MLV003 --skip MLV004 # latency is gated by the load test before release, not per retrain

run_id=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['run_id'])" "$CHARADE_ARTIFACTS_DIR/manifest.json")
aws s3 cp --recursive --only-show-errors "$CHARADE_ARTIFACTS_DIR" "${CHARADE_ARTIFACTS_URI%/}/bundles/$run_id/"
"$(dirname "$0")/promote.sh" "$run_id"
