#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python3}
RESULTS_CSV=${1:?usage: stage2_compute_correlations.sh GARFID_RESULTS_CSV OUT_CSV OUT_TEX}
OUT_CSV=${2:?usage: stage2_compute_correlations.sh GARFID_RESULTS_CSV OUT_CSV OUT_TEX}
OUT_TEX=${3:?usage: stage2_compute_correlations.sh GARFID_RESULTS_CSV OUT_CSV OUT_TEX}
"${PYTHON}" src/scripts/compute_pcc_srcc.py \
  --results-csv "${RESULTS_CSV}" \
  --out-csv "${OUT_CSV}" \
  --out-tex "${OUT_TEX}"
