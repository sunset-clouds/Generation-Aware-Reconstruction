#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
GROUP_ID=${1:?usage: stage2_run_sit_garfid_sweep.sh GROUP_ID IMAGENET_VAL_DIR FID_REF_NPZ OUT_ROOT}
DATASET_DIR=${2:?usage: stage2_run_sit_garfid_sweep.sh GROUP_ID IMAGENET_VAL_DIR FID_REF_NPZ OUT_ROOT}
FID_REF_NPZ=${3:?usage: stage2_run_sit_garfid_sweep.sh GROUP_ID IMAGENET_VAL_DIR FID_REF_NPZ OUT_ROOT}
OUT_ROOT=${4:?usage: stage2_run_sit_garfid_sweep.sh GROUP_ID IMAGENET_VAL_DIR FID_REF_NPZ OUT_ROOT}
ETA_LIST=${ETA_LIST:-0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0}
for eta_t in ${ETA_LIST}; do
  eta_tag=${eta_t//./}
  ETA_T="${eta_t}" bash ./scripts/stage2_run_sit_garfid.sh \
    "${GROUP_ID}" \
    "${DATASET_DIR}" \
    "${FID_REF_NPZ}" \
    "${OUT_ROOT}/eta_${eta_tag}"
done
