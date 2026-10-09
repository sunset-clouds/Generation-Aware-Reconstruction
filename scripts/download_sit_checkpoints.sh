#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python3}

REPO_ID=${SIT_ASSET_REPO_ID:-${STAGE2_ASSET_REPO_ID:-xutongda/Making-rFID-Predictive-of-Diffusion-gFID}}
ASSET_ROOT=${SIT_ASSETS_ROOT:-${STAGE2_ASSETS_ROOT:-assets/sit}}

"${PYTHON}" src/pipelines/download_sit_assets.py \
  --registry_path assets/sit_checkpoint_registry_template.csv \
  --repo_id "${REPO_ID}" \
  --sit_assets_root "${ASSET_ROOT}" \
  "$@"
