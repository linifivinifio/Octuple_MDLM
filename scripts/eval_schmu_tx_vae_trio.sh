#!/bin/bash
#SBATCH --job-name=eval_schmu_tx_trio
#SBATCH --output=logs/eval_schmu_tx_trio_%j.out
#SBATCH --error=logs/eval_schmu_tx_trio_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=student
#SBATCH --account=deep_learning
#SBATCH --gpus=1
#SBATCH --mem=24G

set -euo pipefail

source .venv/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

mkdir -p logs
nvidia-smi || true

MODEL_ID="schmu_tx_vae"
RUN_DIR="runs/schmu_tx_vae_trio"

# Infilling: 50 MIDI files × 2 regions = 100 samples
INFILL_MIDI_DIR="data/test/POP909"

echo "========================================"
echo "Unconditional evaluation"
echo "Model:   ${MODEL_ID}"
echo "Run dir: ${RUN_DIR}"
echo "========================================"

# python3 -m smdiff.cli.evaluate_trio \
#   --task uncond \
#   --model "${MODEL_ID}" \
#   --load_dir "${RUN_DIR}" \
#   --n_samples 100 \
#   --batch_size 4 \
#   --tracks trio

echo "========================================"
echo "Infilling evaluation"
echo "========================================"

python3 -m smdiff.cli.evaluate_trio \
  --task infill \
  --model "${MODEL_ID}" \
  --load_dir "${RUN_DIR}" \
  --input_midi_dir "${INFILL_MIDI_DIR}" \
  --batch_size 4 \
  --mask_token_start 256 \
  --mask_token_end 512 \
  --tracks trio

echo "Job finished at $(date)"
