#!/bin/bash
#SBATCH --job-name=carbon-patches96
#SBATCH --account=brics.u6uc
#SBATCH --partition=workq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/u6uc/asif/logs/carbon-patches-%j.out
#SBATCH --error=/projects/u6uc/asif/logs/carbon-patches-%j.err

set -euo pipefail

export POROUSDIFF_ROOT="${POROUSDIFF_ROOT:-/projects/u6uc/asif}"
CODE_DIR="${CODE_DIR:-${POROUSDIFF_ROOT}/code/PorousDiff}"
VENV_DIR="${VENV_DIR:-${POROUSDIFF_ROOT}/.venv}"

cd "$CODE_DIR"
source "$VENV_DIR/bin/activate"
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"

# Accept positional dataset(s) or DATASETS env var
if [ "$#" -gt 0 ]; then
    DATASET_ARR=("$@")
else
    DATASETS_STR="${DATASETS:-Op008mVV Op018mVV Op033mVV}"
    read -r -a DATASET_ARR <<< "$DATASETS_STR"
fi

VERSION="${VERSION:-carbon_calibrated_v1}"

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Datasets: ${DATASET_ARR[*]}"
echo "Started: $(date -u)"

python -u scripts/preprocessing/extract_carbon_3d_patches.py \
    --datasets "${DATASET_ARR[@]}" \
    --version "$VERSION" \
    --expected-slices "${EXPECTED_SLICES:-0}" \
    --patch-size 96 \
    --stride-xy 96 \
    --stride-z 32 \
    --max-patches "${MAX_PATCHES:-1200}" \
    --min-white 0.005 \
    --max-white 0.90 \
    --val-fraction 0.2 \
    --roi-radius-fraction 0.70 \
    --no-individual-npy \
    --seed 42 \
    --overwrite

echo "Finished: $(date -u)"
