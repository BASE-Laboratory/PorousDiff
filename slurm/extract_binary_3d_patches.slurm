#!/bin/bash
#SBATCH --job-name=op001-binary-patches
#SBATCH --account=brics.u6uc
#SBATCH --partition=workq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/u6uc/asif/logs/binary-patches-%j.out
#SBATCH --error=/projects/u6uc/asif/logs/binary-patches-%j.err

set -euo pipefail

cd /projects/u6uc/asif/code/PorousDiff
source /projects/u6uc/asif/.venv/bin/activate

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Started: $(date)"

python -u scripts/preprocessing/extract_binary_3d_patches.py \
    --dataset Op001 \
    --version v2 \
    --expected-slices 800 \
    --patch-size 64 \
    --stride-xy 128 \
    --stride-z 32 \
    --max-patches 1200 \
    --min-white 0.01 \
    --val-fraction 0.2 \
    --seed 42

echo "Finished: $(date)"
echo "Binary 3D patch extraction completed."