#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=${1:?usage: train_decoder_adaptation.sh CONFIG DATASET_ROOT IMF_CKPT OUT_ROOT}
DATASET_ROOT=${2:?usage: train_decoder_adaptation.sh CONFIG DATASET_ROOT IMF_CKPT OUT_ROOT}
IMF_CKPT=${3:?usage: train_decoder_adaptation.sh CONFIG DATASET_ROOT IMF_CKPT OUT_ROOT}
OUT_ROOT=${4:?usage: train_decoder_adaptation.sh CONFIG DATASET_ROOT IMF_CKPT OUT_ROOT}

mkdir -p "${OUT_ROOT}/checkpoints" "${OUT_ROOT}/results" "${OUT_ROOT}/saver"

torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE:-8}" src/post_train.py \
  --dataset_dir "${DATASET_ROOT}" \
  --pretrained_imf_pytorch "${IMF_CKPT}" \
  --use_pytorch_imf \
  --checkpoint_dir "${OUT_ROOT}/checkpoints" \
  --results_dir "${OUT_ROOT}/results" \
  --saver_dir "${OUT_ROOT}/saver" \
  --yaml_dir "${OUT_ROOT}/resolved_configs" \
  --epochs "${EPOCHS:-10}" \
  --eval_epochs "${EVAL_EPOCHS:-1}" \
  --batch_size "${BATCH_SIZE:-32}" \
  --workers "${NUM_WORKERS:-8}" \
  --max_train_steps "${MAX_TRAIN_STEPS:-0}" \
  --max_eval_steps "${MAX_EVAL_STEPS:-0}" \
  ${EXTRA_ARGS:-}
