#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python3}
DATASET_DIR=${1:?usage: prepare_imagenet_labels.sh IMAGENET_VAL_DIR OUTPUT_LABEL_JSON}
OUTPUT_JSON=${2:?usage: prepare_imagenet_labels.sh IMAGENET_VAL_DIR OUTPUT_LABEL_JSON}
"${PYTHON}" src/scripts/build_imagenet_label_map.py \
  --dataset_dir "${DATASET_DIR}" \
  --output_json "${OUTPUT_JSON}"
