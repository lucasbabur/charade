#!/usr/bin/env bash
# Daily retrain inside the training image (infra/terraform/modules/training_job):
# pull exports -> train -> evaluate -> mlcheck -> promote. Production keeps the previous bundle
# unless every blocking gate passes; WARN-level gates are reported, not blocking.
set -euo pipefail

: "${CHARADE_DATA_URI:?s3:// prefix of impressions.csv and characters.csv}"
: "${CHARADE_ARTIFACTS_URI:?s3:// artifacts bucket}"

export CHARADE_DATA_DIR=/work/data
export CHARADE_ARTIFACTS_DIR=/work/artifacts/current
mkdir -p "$CHARADE_DATA_DIR" "$CHARADE_ARTIFACTS_DIR"

aws s3 cp --only-show-errors "${CHARADE_DATA_URI%/}/impressions.csv" "$CHARADE_DATA_DIR/"
aws s3 cp --only-show-errors "${CHARADE_DATA_URI%/}/characters.csv" "$CHARADE_DATA_DIR/"

poe train
poe ope
poe parity
poe drift
mlcheck . --artifacts-dir "$CHARADE_ARTIFACTS_DIR" --data-dir "$CHARADE_DATA_DIR" \
  --skip MLV003 --skip MLV004  # latency is gated by the load test before release, not per retrain

run_id=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['run_id'])" "$CHARADE_ARTIFACTS_DIR/manifest.json")
aws s3 cp --recursive --only-show-errors "$CHARADE_ARTIFACTS_DIR" "${CHARADE_ARTIFACTS_URI%/}/runs/$run_id/"
aws s3 sync --delete --only-show-errors "$CHARADE_ARTIFACTS_DIR" "${CHARADE_ARTIFACTS_URI%/}/bundles/current/"
echo "promoted $run_id"
