#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python3}
MODEL_TYPE=${1:?usage: evaluate_posttrained_decoder.sh MODEL_TYPE IMF_CKPT DECODER_CKPT IMAGENET_VAL_DIR OUT_DIR REF_NPZ}
IMF_CKPT=${2:?usage: evaluate_posttrained_decoder.sh MODEL_TYPE IMF_CKPT DECODER_CKPT IMAGENET_VAL_DIR OUT_DIR REF_NPZ}
DECODER_CKPT=${3:?usage: evaluate_posttrained_decoder.sh MODEL_TYPE IMF_CKPT DECODER_CKPT IMAGENET_VAL_DIR OUT_DIR REF_NPZ}
DATASET_DIR=${4:?usage: evaluate_posttrained_decoder.sh MODEL_TYPE IMF_CKPT DECODER_CKPT IMAGENET_VAL_DIR OUT_DIR REF_NPZ}
OUT_DIR=${5:?usage: evaluate_posttrained_decoder.sh MODEL_TYPE IMF_CKPT DECODER_CKPT IMAGENET_VAL_DIR OUT_DIR REF_NPZ}
REF_NPZ=${6:?usage: evaluate_posttrained_decoder.sh MODEL_TYPE IMF_CKPT DECODER_CKPT IMAGENET_VAL_DIR OUT_DIR REF_NPZ}
LABEL_MAP_JSON=${LABEL_MAP_JSON:-assets/generated/imagenet_val_dir_to_index.json}
CFG_OMEGA=${CFG_OMEGA:-9.0}
CFG_T_MIN=${CFG_T_MIN:-0.50}
CFG_T_MAX=${CFG_T_MAX:-0.65}
torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC_PER_NODE:-8} src/scripts/generate_eval_images.py \
  --model_type "${MODEL_TYPE}" \
  --pretrained_imf_pytorch "${IMF_CKPT}" \
  --pretrained_decoder "${DECODER_CKPT}" \
  --dataset_dir "${DATASET_DIR}" \
  --label_map_json "${LABEL_MAP_JSON}" \
  --output_dir "${OUT_DIR}" \
  --modes gfid_cfg \
  --num_samples "${NUM_SAMPLES:-50000}" \
  --batch_size "${BATCH_SIZE:-32}" \
  --cfg_omega "${CFG_OMEGA}" \
  --cfg_t_min "${CFG_T_MIN}" \
  --cfg_t_max "${CFG_T_MAX}"
CUDA_VISIBLE_DEVICES="" "${PYTHON}" src/scripts/compute_fid.py \
  --npz_dir "${OUT_DIR}" \
  --model_type "${MODEL_TYPE}" \
  --evaluator openai \
  --ref_npz "${REF_NPZ}" \
  --output_json "${OUT_DIR}/fid_results_openai.json"
