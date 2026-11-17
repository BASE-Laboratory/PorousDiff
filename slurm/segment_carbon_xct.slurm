#!/bin/bash
#SBATCH --job-name=carbon-segment
#SBATCH --account=brics.u6uc
#SBATCH --partition=workq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/projects/u6uc/asif/logs/carbon-segment-%j.out
#SBATCH --error=/projects/u6uc/asif/logs/carbon-segment-%j.err

set -euo pipefail

cd /projects/u6uc/asif/code/PorousDiff
source /projects/u6uc/asif/.venv/bin/activate

echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "Segmentation: Gaussian sigma=50, Otsu threshold"

for dataset in Op008mVV Op018mVV Op033mVV; do
    echo "=================================="
    echo "Processing: $dataset"
    echo "=================================="

    python -u scripts/preprocessing/segment_carbon_xct.py \
        --dataset "$dataset" \
        --expected-slices 800 \
        --sigma 50 \
        --version carbon_sigma50_v1

    echo "Completed: $dataset"
done

echo "All carbon datasets processed."
echo "Finished: $(date)"
