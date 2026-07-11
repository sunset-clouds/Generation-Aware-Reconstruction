#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python3}
GROUP_ID=${1:?usage: stage2_run_sit_garfid.sh GROUP_ID IMAGENET_VAL_DIR FID_REF_NPZ OUT_DIR}
DATASET_DIR=${2:?usage: stage2_run_sit_garfid.sh GROUP_ID IMAGENET_VAL_DIR FID_REF_NPZ OUT_DIR}
FID_REF_NPZ=${3:?usage: stage2_run_sit_garfid.sh GROUP_ID IMAGENET_VAL_DIR FID_REF_NPZ OUT_DIR}
OUT_DIR=${4:?usage: stage2_run_sit_garfid.sh GROUP_ID IMAGENET_VAL_DIR FID_REF_NPZ OUT_DIR}
"${PYTHON}" src/pipelines/run_sit_garfid_single.py \
  --group_id "${GROUP_ID}" \
  --registry_path "${REGISTRY_PATH:-assets/sit_checkpoint_registry_template.csv}" \
  --ifid_repo_root "${IFID_REPO_ROOT:-third_party/ifid}" \
  --dataset_dir "${DATASET_DIR}" \
  --label_map_json "${LABEL_MAP_JSON:-src/scripts/imagenet_val_dir_to_index.json}" \
  --fid_reference_file "${FID_REFERENCE_FILE:-${FID_REF_NPZ}}" \
  --eta_t "${ETA_T:-0.3}" \
  --num_samples "${NUM_SAMPLES:-50000}" \
  --batch_size "${BATCH_SIZE:-16}" \
  --num_workers "${NUM_WORKERS:-0}" \
  --output_dir "${OUT_DIR}" \
  ${USE_CFG:+--use_cfg} \
  ${SAVE_NPZ:+--save_npz}
