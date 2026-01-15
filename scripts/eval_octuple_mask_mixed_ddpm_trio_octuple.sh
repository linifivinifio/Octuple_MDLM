#!/bin/bash
#SBATCH --job-name=eval_mixed
#SBATCH --output=logs/eval_mixed_%j.out
#SBATCH --error=logs/eval_mixed_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=student
#SBATCH --account=deep_learning
#SBATCH --gpus=1
#SBATCH --mem=24G

set -euo pipefail

source .venv/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

mkdir -p logs
nvidia-smi || true

MODEL_ID="octuple_mask_ddpm"
RUN_DIR="runs/octuple_mask_ddpm_trio_octuple_mixed"
INFILL_MIDI_DIR="data/test/POP909"

echo "========================================"
echo "Unconditional evaluation"
echo "Model:   ${MODEL_ID}"
echo "Run dir: ${RUN_DIR}"
echo "Strategy: mixed"
echo "========================================"

python3 -m smdiff.cli.evaluate_octuple \
  --task uncond \
  --model "${MODEL_ID}" \
  --load_dir "${RUN_DIR}" \
  --n_samples 100 \
  --batch_size 4

echo "========================================"
echo "Infilling evaluation"
echo "MIDI dir: ${INFILL_MIDI_DIR}"
echo "========================================"

python3 -m smdiff.cli.evaluate_octuple \
  --task infill \
  --model "${MODEL_ID}" \
  --load_dir "${RUN_DIR}" \
  --input_midi_dir "${INFILL_MIDI_DIR}" \
  --batch_size 4 \
  --mask_token_start 256 \
  --mask_token_end 512

echo "Job finished at $(date)"
