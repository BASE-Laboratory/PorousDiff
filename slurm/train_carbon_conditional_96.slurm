#!/bin/bash
#SBATCH --job-name=carbon-cond-96
#SBATCH --account=brics.u6uc
#SBATCH --partition=workq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/projects/u6uc/asif/logs/carbon-cond-96-%j.out
#SBATCH --error=/projects/u6uc/asif/logs/carbon-cond-96-%j.err

set -euo pipefail

export POROUSDIFF_ROOT="${POROUSDIFF_ROOT:-/projects/u6uc/asif}"
CODE_DIR="${CODE_DIR:-${POROUSDIFF_ROOT}/code/PorousDiff}"
VENV_DIR="${VENV_DIR:-${POROUSDIFF_ROOT}/.venv}"

cd "$CODE_DIR"
source "$VENV_DIR/bin/activate"
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"

EXPERIMENT="${EXPERIMENT:-carbon_conditional_96_v2}"
VERSION="${VERSION:-carbon_calibrated_v1}"
EPOCHS="${EPOCHS:-100}"
STATES_STR="${STATES:-Op008mVV=0.008 Op018mVV=0.018 Op033mVV=0.033}"
read -r -a STATES_ARR <<< "$STATES_STR"

RESUME_ARGS=()
if [ -f "${POROUSDIFF_ROOT}/checkpoints/${EXPERIMENT}/latest.pth" ]; then
    echo "Existing checkpoint detected; resuming automatically."
    RESUME_ARGS+=(--resume)
fi

echo "Started: $(date -u)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Experiment: ${EXPERIMENT} (target epochs=${EPOCHS})"
echo "Training states: ${STATES_ARR[*]}"
nvidia-smi

python -u scripts/training/train_carbon_conditional_96.py \
    --experiment "$EXPERIMENT" \
    --version "$VERSION" \
    --states "${STATES_ARR[@]}" \
    --patch-size "${PATCH_SIZE:-96}" \
    --channels "${CHANNELS:-32,64,128}" \
    --schedule "${SCHEDULE:-zero_snr_scaled_linear}" \
    --prediction-type "${PREDICTION_TYPE:-v_prediction}" \
    --conditioning-mode "${CONDITIONING_MODE:-multiscale_fourier}" \
    --epochs "$EPOCHS" \
    --batch-size "${BATCH_SIZE:-4}" \
    --grad-accum "${GRAD_ACCUM:-2}" \
    --lr "${LR:-2e-4}" \
    --amp bf16 \
    --num-workers 4 \
    --save-every 10 \
    --seed "${SEED:-42}" \
    "${RESUME_ARGS[@]}"

echo "Finished: $(date -u)"
