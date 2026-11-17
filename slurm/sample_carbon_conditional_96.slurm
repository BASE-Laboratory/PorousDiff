#!/bin/bash
#SBATCH --job-name=carbon-sample-96
#SBATCH --account=brics.u6uc
#SBATCH --partition=workq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/u6uc/asif/logs/carbon-sample-96-%j.out
#SBATCH --error=/projects/u6uc/asif/logs/carbon-sample-96-%j.err

set -euo pipefail

export POROUSDIFF_ROOT="${POROUSDIFF_ROOT:-/projects/u6uc/asif}"
CODE_DIR="${CODE_DIR:-${POROUSDIFF_ROOT}/code/PorousDiff}"
VENV_DIR="${VENV_DIR:-${POROUSDIFF_ROOT}/.venv}"

cd "$CODE_DIR"
source "$VENV_DIR/bin/activate"
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"

EXPERIMENT="${EXPERIMENT:-carbon_conditional_96_v2}"
VERSION="${VERSION:-carbon_calibrated_v1}"
EVAL_STATES_STR="${EVAL_STATES:-Op008mVV=0.008 Op018mVV=0.018 Op033mVV=0.033}"
read -r -a EVAL_STATES_ARR <<< "$EVAL_STATES_STR"
HELDOUT_STR="${HELDOUT_STATES:-}"
read -r -a HELDOUT_ARR <<< "$HELDOUT_STR"
SWEEP_LOADS="${SWEEP_LOADS:-0.008,0.013,0.018,0.023,0.028,0.033,0.038,0.043,0.048,0.053}"

echo "Started: $(date -u)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
nvidia-smi

python -u scripts/evaluation/sample_carbon_conditional_96.py \
    --experiment "$EXPERIMENT" \
    --sampler "${SAMPLER:-ddim}" \
    --steps "${STEPS:-100}" \
    --guidance-scale "${GUIDANCE_SCALE:-1.5}" \
    --num-samples "${NUM_SAMPLES:-20}" \
    --batch-size "${BATCH_SIZE:-4}" \
    --eval-states "${EVAL_STATES_ARR[@]}" \
    --sweep-loads "$SWEEP_LOADS" \
    --export-openimpala

HELDOUT_ARGS=()
if [ "${#HELDOUT_ARR[@]}" -gt 0 ]; then
    HELDOUT_ARGS+=(--heldout-states "${HELDOUT_ARR[@]}")
fi

python -u scripts/evaluation/evaluate_carbon_physics.py \
    --experiment "$EXPERIMENT" \
    --version "$VERSION" \
    --states "${EVAL_STATES_ARR[@]}" \
    "${HELDOUT_ARGS[@]}"

echo "Finished: $(date -u)"
