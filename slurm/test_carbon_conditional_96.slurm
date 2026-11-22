#!/bin/bash
#SBATCH --job-name=carbon-cond-test
#SBATCH --account=brics.u6uc
#SBATCH --partition=workq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/projects/u6uc/asif/logs/carbon-cond-test-%j.out
#SBATCH --error=/projects/u6uc/asif/logs/carbon-cond-test-%j.err

set -euo pipefail

export POROUSDIFF_ROOT="${POROUSDIFF_ROOT:-/projects/u6uc/asif}"
CODE_DIR="${CODE_DIR:-${POROUSDIFF_ROOT}/code/PorousDiff}"
VENV_DIR="${VENV_DIR:-${POROUSDIFF_ROOT}/.venv}"

cd "$CODE_DIR"
source "$VENV_DIR/bin/activate"
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"

STATES_STR="${STATES:-Op008mVV=0.008 Op018mVV=0.018 Op033mVV=0.033}"
read -r -a STATES_ARR <<< "$STATES_STR"
VERSION="${VERSION:-carbon_calibrated_v1}"

echo "Started: $(date -u)"
nvidia-smi

python -u scripts/training/train_carbon_conditional_96.py \
    --experiment carbon_conditional_96_smoke \
    --version "$VERSION" \
    --states "${STATES_ARR[@]}" \
    --epochs 1 \
    --batch-size 2 \
    --grad-accum 1 \
    --amp bf16 \
    --max-steps 4 \
    --max-val-steps 2

echo "Finished: $(date -u)"
