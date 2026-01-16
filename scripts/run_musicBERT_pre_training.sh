#!/bin/bash
#SBATCH --job-name=MusicBERT_small
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=student 
#SBATCH --account deep_learning

# Initialize Conda (Assuming standard installation path, adjust if needed)
# source $HOME/miniconda3/etc/profile.d/conda.sh
# conda activate symbolic-music-discrete-diffusion

# OR if using modules on the cluster:

. /home/lziltener/jupyter/bin/activate
source .venv/bin/activate
nvidia-smi
echo "Starting training job on $HOSTNAME"
echo "Date: $(date)"

# Ensure logs directory exists
mkdir -p logs

# Run the training script
python3 src/smdiff/models/PT_encoder_musicBERT/train_musicBERT.py

echo "Job finished at $(date)"