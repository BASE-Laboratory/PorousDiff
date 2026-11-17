#!/bin/bash
#SBATCH --job-name=carbon-cond-100
#SBATCH --account=brics.u6uc
#SBATCH --partition=workq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/projects/u6uc/asif/logs/carbon-cond-100-%j.out
#SBATCH --error=/projects/u6uc/asif/logs/carbon-cond-100-%j.err

set -euo pipefail

cd /projects/u6uc/asif/code/PorousDiff
source /projects/u6uc/asif/.venv/bin/activate

echo "Started: $(date -u)"
echo "Job ID: ${SLURM_JOB_ID}"

python -u scripts/training/train_carbon_conditional_96.py \
    --experiment carbon_conditional_96_v1 \
    --epochs 28 \
    --batch-size 1 \
    --lr 1e-4 \
    --seed 42 \
    --resume

echo "Finished: $(date -u)"
