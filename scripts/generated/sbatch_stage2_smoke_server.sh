#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --job-name=garfid_s2_smoke
#SBATCH --chdir=.
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

BASE_DIR=${BASE_DIR:-${HOME}/garfid_data}
PROJECT_DIR=${PROJECT_DIR:-$(pwd)}
VENV_DIR=${VENV_DIR:-${HOME}/.venvs/garfid}
IFID_REPO=${IFID_REPO:-${PROJECT_DIR}/third_party/ifid}
DATASET_VAL=${DATASET_VAL:-${BASE_DIR}/dataset/imagenet/val}
LABEL_MAP_JSON=${LABEL_MAP_JSON:-${PROJECT_DIR}/assets/generated/imagenet_val_dir_to_index.json}
SDVAE_CKPT=${SDVAE_CKPT:-${BASE_DIR}/vae_assets/sdvae/sdvae-f8d4.pt}
OUT_DIR=${OUT_DIR:-${PROJECT_DIR}/results/server_smoke/stage2_sdvae_b_${SLURM_JOB_ID:-manual}}

source "${VENV_DIR}/bin/activate"
cd "${PROJECT_DIR}"

export HF_HOME=${BASE_DIR}/hf_cache
export TORCH_HOME=${BASE_DIR}/torch_cache
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export SDVAE_CKPT
export MPLCONFIGDIR=${BASE_DIR}/matplotlib_cache
export PYTHONUNBUFFERED=1

mkdir -p "${OUT_DIR}" "${PROJECT_DIR}/logs/slurm" "${PROJECT_DIR}/assets/generated" "${MPLCONFIGDIR}"

echo "============================================"
echo " GAR-FID Stage2 smoke"
echo " Job ID: ${SLURM_JOB_ID:-manual}"
echo " Node:   ${SLURM_NODELIST:-unknown}"
echo " Start:  $(date)"
echo "============================================"
nvidia-smi -L || true

if [[ ! -f "${LABEL_MAP_JSON}" ]]; then
  bash scripts/prepare_imagenet_labels.sh "${DATASET_VAL}" "${LABEL_MAP_JSON}"
fi

if [[ ! -f "${PROJECT_DIR}/assets/stage2_assets/sit-b-sdvae-400k/args.json" || \
      ! -f "${PROJECT_DIR}/assets/stage2_assets/sit-b-sdvae-400k/checkpoints/0400000.pt" ]]; then
  HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
    bash scripts/stage2_download_ifid_assets.sh --group_ids sdvae_b
fi

python -u src/pipelines/smoke_test_sit_gar.py \
  --group_id sdvae_b \
  --registry_path assets/sit_checkpoint_registry_template.csv \
  --ifid_repo_root "${IFID_REPO}" \
  --dataset_dir "${DATASET_VAL}" \
  --label_map_json "${LABEL_MAP_JSON}" \
  --output_dir "${OUT_DIR}" \
  --num_images 2 \
  --eta_t 0.3 \
  --num_steps 4 \
  --sampling_mode sde \
  --step_policy fixed \
  --device cuda:0

echo "============================================"
echo " Stage2 smoke complete"
echo " End: $(date)"
echo " Output: ${OUT_DIR}"
echo "============================================"
