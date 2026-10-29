#!/bin/bash
#SBATCH --job-name=op001-sample96
#SBATCH --account=brics.u6uc
#SBATCH --partition=workq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/u6uc/asif/logs/sample-binary-96-%j.out
#SBATCH --error=/projects/u6uc/asif/logs/sample-binary-96-%j.err

set -euo pipefail

cd /projects/u6uc/asif/code/PorousDiff
source /projects/u6uc/asif/.venv/bin/activate

echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"

python -u scripts/evaluation/sample_binary_diffusion_96.py \
    --epoch 100 \
    --num-samples 10 \
    --seed 42

echo "Finished: $(date)"
echo "96³ sampling completed."
