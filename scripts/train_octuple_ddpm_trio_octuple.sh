#!/bin/bash
#SBATCH --job-name=oct_ddpm_trio
#SBATCH --output=logs/oct_ddpm_trio_%j.out
#SBATCH --error=logs/oct_ddpm_trio_%j.err
#SBATCH --time=16:00:00
#SBATCH --partition=student
#SBATCH --account=deep_learning
#SBATCH --gpus=1
#SBATCH --mem=24G

set -euo pipefail

source .venv/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

mkdir -p logs
nvidia-smi || true


echo "Job finished at $(date)"
