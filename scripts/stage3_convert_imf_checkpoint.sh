#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python3}

MODEL_TYPE=${1:?usage: convert_imf_checkpoint.sh MODEL_TYPE JAX_CHECKPOINT_DIR OUTPUT_PT}
JAX_CHECKPOINT_DIR=${2:?usage: convert_imf_checkpoint.sh MODEL_TYPE JAX_CHECKPOINT_DIR OUTPUT_PT}
OUTPUT_PT=${3:?usage: convert_imf_checkpoint.sh MODEL_TYPE JAX_CHECKPOINT_DIR OUTPUT_PT}

"${PYTHON}" src/tools/convert_jax_to_pytorch.py \
  --model_type "${MODEL_TYPE}" \
  --jax_checkpoint "${JAX_CHECKPOINT_DIR}" \
  --output_path "${OUTPUT_PT}"
