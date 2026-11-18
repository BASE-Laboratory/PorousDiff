#!/bin/bash
#SBATCH --job-name=carbon-cond-bench
#SBATCH --account=brics.u6uc
#SBATCH --partition=workq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/projects/u6uc/asif/logs/carbon-cond-bench-%j.out
#SBATCH --error=/projects/u6uc/asif/logs/carbon-cond-bench-%j.err

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
    --experiment carbon_conditional_96_benchmark \
    --version "$VERSION" \
    --states "${STATES_ARR[@]}" \
    --epochs 1 \
    --batch-size 4 \
    --grad-accum 2 \
    --amp bf16 \
    --num-workers 4 \
    --max-steps 100 \
    --max-val-steps 20

echo "Finished: $(date -u)"
