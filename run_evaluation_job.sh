#!/bin/bash
#SBATCH --job-name=eval_ddpm
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=student 
#SBATCH --account=deep_learning

set -euo pipefail

source .venv/bin/activate

# Ensure Python can import project packages when running scripts by path
# Prefer src/ on PYTHONPATH so 'smdiff' resolves
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

nvidia-smi || true
echo "Starting evaluation on $(hostname)"
echo "Date: $(date)"

# Ensure logs directory exists
mkdir -p logs

# Sanity: show Python and torch CUDA availability
python -V || python3 -V
python -c "import torch; print('Torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())" || true

# Configuration
MODEL_ID="schmu_tx_vae"
RUN_DIR="runs/schmu_tx_vae_trio"  # Update to your actual run directory
DATASET_ID="pop909_trio"

# Mini smoke-test sizes (fast). Increase once the job works end-to-end.
N_SAMPLES_UNCOND=8
INFILL_MIDI_DIR="data/test/POP909"  # recursively searched

# Navigate to repository (prefer SLURM submission directory)
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_DIR"

echo "========================================"
echo "Starting Model Evaluation"
echo "Model: $MODEL_ID"
echo "Run Dir: $RUN_DIR"
echo "Dataset: $DATASET_ID"
echo "========================================"

# ============================================================
# EXAMPLE 1: UNCONDITIONAL EVALUATION - GENERATE NEW SAMPLES
# ============================================================
# This configuration generates new samples during evaluation
# Use when: You want fresh samples with specific sampling parameters
echo ""
echo "Running unconditional evaluation (generating new samples)..."
python evaluate_trio.py \
    --task uncond \
    --model $MODEL_ID \
    --load_dir $RUN_DIR \
    --n_samples $N_SAMPLES_UNCOND \
    --batch_size 4

echo "Unconditional evaluation complete!"

# ============================================================
# EXAMPLE 2: INFILLING EVALUATION - GENERATE CONDITIONED SAMPLES
# ============================================================
# This configuration generates conditioned samples directly from MIDIs.
# It will recursively scan --input_midi_dir.
echo ""
echo "Running infilling evaluation (generating conditioned samples)..."

python evaluate_trio.py \
    --task infill \
    --model $MODEL_ID \
    --load_dir $RUN_DIR \
    --input_midi_dir "$INFILL_MIDI_DIR" \
    --batch_size 4 \
    --n_midis 2 \
    --mask_token_start 256 \
    --mask_token_end 512

echo "Infilling evaluation complete!"

echo ""
echo "========================================"
echo "All evaluations complete!"
echo "Results saved to: $RUN_DIR/metrics/"
echo "========================================"
