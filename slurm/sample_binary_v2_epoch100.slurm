#!/bin/bash
#SBATCH --job-name=op001-binary-v2-sample100
#SBATCH --account=brics.u6uc
#SBATCH --partition=workq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/projects/u6uc/asif/logs/sample-binary-v2-%j.out
#SBATCH --error=/projects/u6uc/asif/logs/sample-binary-v2-%j.err

set -euo pipefail

cd /projects/u6uc/asif/code/PorousDiff
source /projects/u6uc/asif/.venv/bin/activate

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"

nvidia-smi

python -u scripts/evaluation/sample_binary_diffusion_v2.py \
    --epoch 100 \
    --num-samples 10 \
    --seed 42

echo "Finished: $(date)"
echo "Binary DDPM v2 sampling completed."
