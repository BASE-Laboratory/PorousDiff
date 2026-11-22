#!/bin/bash
#SBATCH --job-name=carbon-segment
#SBATCH --account=brics.u6uc
#SBATCH --partition=workq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/projects/u6uc/asif/logs/carbon-segment-%j.out
#SBATCH --error=/projects/u6uc/asif/logs/carbon-segment-%j.err

set -euo pipefail

export POROUSDIFF_ROOT="${POROUSDIFF_ROOT:-/projects/u6uc/asif}"
CODE_DIR="${CODE_DIR:-${POROUSDIFF_ROOT}/code/PorousDiff}"
VENV_DIR="${VENV_DIR:-${POROUSDIFF_ROOT}/.venv}"

cd "$CODE_DIR"
source "$VENV_DIR/bin/activate"
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"

# Override via `sbatch --export=ALL,DATASETS="0p008mVV 0p013mVV ..."` when all loads are staged
DATASETS_STR="${DATASETS:-Op008mVV Op018mVV Op033mVV}"
read -r -a DATASET_ARR <<< "$DATASETS_STR"
VERSION="${VERSION:-carbon_calibrated_v1}"
DELTA_THRESHOLD="${DELTA_THRESHOLD:-5500}"

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Started: $(date -u)"
echo "Datasets: ${DATASET_ARR[*]}"
echo "Version: ${VERSION} (delta_threshold=${DELTA_THRESHOLD}, circular ROI + Z-artifact gating)"

python -u scripts/preprocessing/segment_carbon_xct.py \
    --datasets "${DATASET_ARR[@]}" \
    --version "$VERSION" \
    --delta-threshold "$DELTA_THRESHOLD" \
    --sigma 32 \
    --roi-radius-fraction 0.70 \
    --expected-slices "${EXPECTED_SLICES:-0}" \
    --overwrite

echo "All carbon datasets segmented successfully."
echo "Finished: $(date -u)"
