#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python3}
GROUP_ID=${1:?usage: stage2_smoke_test.sh GROUP_ID IMAGENET_VAL_DIR}
DATASET_DIR=${2:?usage: stage2_smoke_test.sh GROUP_ID IMAGENET_VAL_DIR}
"${PYTHON}" src/pipelines/smoke_test_sit_gar.py \
  --group_id "${GROUP_ID}" \
  --registry_path assets/sit_checkpoint_registry_template.csv \
  --ifid_repo_root third_party/ifid \
  --dataset_dir "${DATASET_DIR}" \
  --label_map_json src/scripts/imagenet_val_dir_to_index.json \
  --output_dir "results/stage2_smoke/${GROUP_ID}"
