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

python3 src/smdiff/cli/train.py \
  --model octuple_ddpm \
  --dataset_id pop909_trio_octuple \
  --batch_size 4 \
  --epochs 100 \
  --steps_per_log 10 \
  --steps_per_eval 1000 \
  --steps_per_sample 5000 \
  --steps_per_checkpoint 5000 \
  --seed 67 \
  --wandb \
  --wandb_project "octubert-music" \
  --wandb_name "octuple-ddpm-trio-octuple"

echo "Job finished at $(date)"
