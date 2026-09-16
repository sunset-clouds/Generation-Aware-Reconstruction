#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --job-name=garfid_s3_smoke
#SBATCH --chdir=.
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

BASE_DIR=${BASE_DIR:-${HOME}/garfid_data}
PROJECT_DIR=${PROJECT_DIR:-$(pwd)}
VENV_DIR=${VENV_DIR:-${HOME}/.venvs/garfid}
DATASET_ROOT=${DATASET_ROOT:-${BASE_DIR}/dataset/imagenet_idx}
IMF_CKPT=${IMF_CKPT:-${BASE_DIR}/IMF-GAP_V1/checkpoints/pytorch/iMF-B-2.pt}
LPIPS_VGG_CKPT=${LPIPS_VGG_CKPT:-${BASE_DIR}/IMF-GAP_V1/models/cache/vgg.pth}
OUT_ROOT=${OUT_ROOT:-${PROJECT_DIR}/results/server_smoke/stage3_b2_${SLURM_JOB_ID:-manual}}

source "${VENV_DIR}/bin/activate"
cd "${PROJECT_DIR}"

export HF_HOME=${BASE_DIR}/hf_cache
export TORCH_HOME=${BASE_DIR}/torch_cache
export SD_VAE_PATH=${SD_VAE_PATH:-${BASE_DIR}/vae_assets/sd-vae-ft-mse}
export LPIPS_VGG_CKPT
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export MPLCONFIGDIR=${BASE_DIR}/matplotlib_cache
export PYTHONUNBUFFERED=1
export NCCL_TIMEOUT=3600
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

mkdir -p "${OUT_ROOT}" "${PROJECT_DIR}/logs/slurm" "${MPLCONFIGDIR}"

SMOKE_EXTRA_ARGS=${SMOKE_EXTRA_ARGS:-"--model_type iMF-B-2 --disable_gan --fixed_noise --maximum_noise_level 0.4 --disc_start_epoch 99"}

echo "============================================"
echo " GAR-FID Stage3 smoke"
echo " Job ID: ${SLURM_JOB_ID:-manual}"
echo " Node:   ${SLURM_NODELIST:-unknown}"
echo " Start:  $(date)"
echo "============================================"
nvidia-smi -L || true

NPROC_PER_NODE=1 \
EPOCHS=1 \
EVAL_EPOCHS=999 \
MAX_TRAIN_STEPS=2 \
MAX_EVAL_STEPS=0 \
BATCH_SIZE=2 \
NUM_WORKERS=2 \
EXTRA_ARGS="${SMOKE_EXTRA_ARGS}" \
bash scripts/stage3_train_decoder_adaptation.sh \
  "${DATASET_ROOT}" \
  "${IMF_CKPT}" \
  "${OUT_ROOT}"

echo "============================================"
echo " Stage3 smoke complete"
echo " End: $(date)"
echo " Output: ${OUT_ROOT}"
echo "============================================"
