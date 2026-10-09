#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --job-name=decoder_adaptation_smoke
#SBATCH --chdir=.
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
# Compatibility entrypoint; use sbatch_decoder_adaptation_smoke_server.sh.
set -euo pipefail
exec bash "${PROJECT_DIR:-${PWD}}/scripts/generated/sbatch_decoder_adaptation_smoke_server.sh" "$@"
