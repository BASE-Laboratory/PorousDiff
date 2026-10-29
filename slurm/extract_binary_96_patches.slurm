#!/bin/bash
#SBATCH --job-name=op001-binary-96
#SBATCH --account=brics.u6uc
#SBATCH --partition=workq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/u6uc/asif/logs/op001-patches96-%j.out
#SBATCH --error=/projects/u6uc/asif/logs/op001-patches96-%j.err

set -euo pipefail

cd /projects/u6uc/asif/code/PorousDiff
source /projects/u6uc/asif/.venv/bin/activate

echo "Started: $(date)"

python -u scripts/preprocessing/extract_binary_3d_patches.py \
    --dataset Op001 \
    --version v2 \
    --expected-slices 800 \
    --patch-size 96 \
    --stride-xy 128 \
    --stride-z 32 \
    --max-patches 1200 \
    --min-white 0.01 \
    --max-white 0.99 \
    --val-fraction 0.2 \
    --seed 42

echo "Finished: $(date)"
