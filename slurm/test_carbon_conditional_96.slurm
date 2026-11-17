#!/bin/bash
#SBATCH --job-name=carbon-cond-test
#SBATCH --account=brics.u6uc
#SBATCH --partition=workq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/projects/u6uc/asif/logs/carbon-cond-test-%j.out
#SBATCH --error=/projects/u6uc/asif/logs/carbon-cond-test-%j.err

set -euo pipefail

cd /projects/u6uc/asif/code/PorousDiff
source /projects/u6uc/asif/.venv/bin/activate

echo "Started: $(date -u)"
nvidia-smi

python -u scripts/training/train_carbon_conditional_96.py \
    --experiment carbon_conditional_96_smoke \
    --epochs 1 \
    --batch-size 1 \
    --max-steps 3 \
    --max-val-steps 2

echo "Finished: $(date -u)"
