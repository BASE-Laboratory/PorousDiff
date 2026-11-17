#!/bin/bash
#SBATCH --job-name=carbon-patches96
#SBATCH --account=brics.u6uc
#SBATCH --partition=workq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/u6uc/asif/logs/carbon-patches-%j.out
#SBATCH --error=/projects/u6uc/asif/logs/carbon-patches-%j.err

set -euo pipefail

cd /projects/u6uc/asif/code/PorousDiff
source /projects/u6uc/asif/.venv/bin/activate

DATASET="${1:?Provide dataset name}"

echo "Dataset: $DATASET"
echo "Started: $(date)"

python -u scripts/preprocessing/extract_carbon_3d_patches.py \
    --dataset "$DATASET" \
    --version carbon_sigma50_v1 \
    --expected-slices 800 \
    --patch-size 96 \
    --stride-xy 128 \
    --stride-z 32 \
    --max-patches 1200 \
    --min-white 0.0 \
    --max-white 1.0 \
    --val-fraction 0.2 \
    --seed 42

echo "Finished: $(date)"
