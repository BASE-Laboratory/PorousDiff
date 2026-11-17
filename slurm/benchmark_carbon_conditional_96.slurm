#!/bin/bash
#SBATCH --job-name=carbon-cond-bench
#SBATCH --account=brics.u6uc
#SBATCH --partition=workq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/projects/u6uc/asif/logs/carbon-cond-bench-%j.out
#SBATCH --error=/projects/u6uc/asif/logs/carbon-cond-bench-%j.err

set -euo pipefail

cd /projects/u6uc/asif/code/PorousDiff
source /projects/u6uc/asif/.venv/bin/activate

echo "Started: $(date -u)"
nvidia-smi

python -u scripts/training/train_carbon_conditional_96.py \
    --experiment carbon_conditional_96_benchmark \
    --epochs 1 \
    --batch-size 1 \
    --max-steps 100 \
    --max-val-steps 20

echo "Finished: $(date -u)"
