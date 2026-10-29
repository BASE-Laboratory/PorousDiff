#!/bin/bash
#SBATCH --job-name=segment-binary-xct
#SBATCH --account=brics.u6uc
#SBATCH --partition=workq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/projects/u6uc/asif/logs/segment-binary-%j.out
#SBATCH --error=/projects/u6uc/asif/logs/segment-binary-%j.err

set -euo pipefail

cd /projects/u6uc/asif/code/PorousDiff
source /projects/u6uc/asif/.venv/bin/activate

DATASET="${1:?Provide dataset name}"
SLICES="${2:?Provide expected slice count}"
VERSION="${3:-v2}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Dataset: ${DATASET}"
echo "Expected slices: ${SLICES}"
echo "Version: ${VERSION}"
echo "Started: $(date)"

python -u scripts/preprocessing/segment_binary_xct.py \
    --dataset "$DATASET" \
    --expected-slices "$SLICES" \
    --version "$VERSION"

echo "Finished: $(date)"
echo "Binary segmentation completed."
