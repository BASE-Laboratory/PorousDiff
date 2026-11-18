#!/bin/bash
#SBATCH --job-name=beamtime-mult
#SBATCH --account=brics.u6uc
#SBATCH --partition=workq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/projects/u6uc/asif/logs/beamtime-mult-%j.out
#SBATCH --error=/projects/u6uc/asif/logs/beamtime-mult-%j.err

set -euo pipefail

export POROUSDIFF_ROOT="${POROUSDIFF_ROOT:-/projects/u6uc/asif}"
CODE_DIR="${CODE_DIR:-${POROUSDIFF_ROOT}/code/PorousDiff}"
VENV_DIR="${VENV_DIR:-${POROUSDIFF_ROOT}/.venv}"
DATA_ROOT="${DATA_ROOT:-${POROUSDIFF_ROOT}/data/carbon_virgin}"
MANIFEST="${MANIFEST:-${DATA_ROOT}/bundle_manifest.json}"

cd "$CODE_DIR"
source "$VENV_DIR/bin/activate"
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"

EXPERIMENT="${EXPERIMENT:-beamtime_multiplier_96}"
VERSION="${VERSION:-carbon_calibrated_v1}"
EPOCHS="${EPOCHS:-100}"
PATCH_SIZE="${PATCH_SIZE:-96}"

INCLUDE_ARGS=()
if [ -n "${INCLUDE_STATES:-}" ]; then
    read -r -a INC_ARR <<< "$INCLUDE_STATES"
    INCLUDE_ARGS+=(--include-states "${INC_ARR[@]}")
fi

EXCLUDE_ARGS=()
HELDOUT_ARGS=()
if [ -n "${HELDOUT_STATES:-}" ]; then
    read -r -a EXC_ARR <<< "$HELDOUT_STATES"
    EXCLUDE_ARGS+=(--exclude-states "${EXC_ARR[@]}")
    HELDOUT_ARGS+=(--heldout-states "${EXC_ARR[@]}")
fi

RESUME_ARGS=()
if [ -f "${POROUSDIFF_ROOT}/checkpoints/${EXPERIMENT}/latest.pth" ]; then
    echo "Existing checkpoint detected; resuming automatically."
    RESUME_ARGS+=(--resume)
fi

PHYSICS_ARGS=()
if [ "${PHYSICS_LOSS:-1}" = "1" ]; then
    PHYSICS_ARGS+=(--physics-loss)
fi

STRESS_ARGS=()
if [ "${USE_STRESS_KPA:-auto}" = "1" ]; then
    STRESS_ARGS+=(--use-stress-kpa)
elif [ "${USE_STRESS_KPA:-auto}" = "auto" ]; then
    if [ -f "${POROUSDIFF_ROOT}/checkpoints/${EXPERIMENT}/latest.pth" ]; then
        if python -c "import sys, torch; c = torch.load(sys.argv[1], map_location='cpu', weights_only=False).get('config', {}); sys.exit(0 if c.get('use_stress_kpa', False) else 1)" "${POROUSDIFF_ROOT}/checkpoints/${EXPERIMENT}/latest.pth"; then
            STRESS_ARGS+=(--use-stress-kpa)
        fi
    elif [ -f "$MANIFEST" ]; then
        if python -c "import json, sys; s = json.load(open(sys.argv[1])).get('scans', []); sys.exit(0 if (s and all(x.get('stress_kpa') is not None for x in s)) else 1)" "$MANIFEST"; then
            echo "All manifest scans carry calibrated stress_kpa; enabling --use-stress-kpa."
            STRESS_ARGS+=(--use-stress-kpa)
        fi
    fi
fi

echo "Started Beamtime Multiplier pipeline: $(date -u)"
echo "Manifest: $MANIFEST | Experiment: $EXPERIMENT"
nvidia-smi

python -u scripts/training/train_carbon_conditional_96.py \
    --experiment "$EXPERIMENT" \
    --data-root "$DATA_ROOT" \
    --manifest-json "$MANIFEST" \
    --version "$VERSION" \
    --patch-size "$PATCH_SIZE" \
    --balanced-sampling \
    --epochs "$EPOCHS" \
    --batch-size "${BATCH_SIZE:-4}" \
    --grad-accum "${GRAD_ACCUM:-2}" \
    --lr "${LR:-2e-4}" \
    --amp bf16 \
    --num-workers 4 \
    "${STRESS_ARGS[@]}" \
    "${PHYSICS_ARGS[@]}" \
    "${INCLUDE_ARGS[@]}" \
    "${EXCLUDE_ARGS[@]}" \
    "${RESUME_ARGS[@]}"

python -u scripts/evaluation/sample_carbon_conditional_96.py \
    --experiment "$EXPERIMENT" \
    --manifest-json "$MANIFEST" \
    --sampler "${SAMPLER:-ddim}" \
    --steps "${STEPS:-100}" \
    --guidance-scale "${GUIDANCE_SCALE:-1.5}" \
    --num-samples "${NUM_SAMPLES:-20}" \
    --batch-size "${BATCH_SIZE:-4}" \
    --sweep-loads "" \
    --export-openimpala

python -u scripts/evaluation/evaluate_beamtime_multiplier.py \
    --experiments "porousdiff=${EXPERIMENT}" \
    --data-root "$DATA_ROOT" \
    --manifest-json "$MANIFEST" \
    --version "$VERSION" \
    --patch-size "$PATCH_SIZE" \
    --include-virtual-compression \
    --generate-openimpala-suite \
    --export-real-openimpala \
    "${HELDOUT_ARGS[@]}"

python -u scripts/evaluation/optimize_clamping_pressure.py \
    --experiments "porousdiff=${EXPERIMENT}" \
    --data-root "$DATA_ROOT" \
    --manifest-json "$MANIFEST" \
    --version "$VERSION" \
    --patch-size "$PATCH_SIZE" \
    --dense-sweep-points "${DENSE_SWEEP_POINTS:-15}"

echo "Finished: $(date -u)"
