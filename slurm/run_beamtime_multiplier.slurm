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
#SBATCH --time=12:00:00
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
SEEDS="${SEEDS:-42 43 44}"
read -r -a SEED_ARR <<< "$SEEDS"

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

PHYSICS_ARGS=()
if [ "${PHYSICS_LOSS:-1}" = "1" ]; then
    PHYSICS_ARGS+=(--physics-loss)
fi

FIRST_SEED_EXP="$EXPERIMENT"
if [ "${#SEED_ARR[@]}" -gt 1 ]; then
    FIRST_SEED_EXP="${EXPERIMENT}_s${SEED_ARR[0]}"
fi

STRESS_ARGS=()
if [ "${USE_STRESS_KPA:-auto}" = "1" ]; then
    STRESS_ARGS+=(--use-stress-kpa)
elif [ "${USE_STRESS_KPA:-auto}" = "auto" ]; then
    if [ -f "${POROUSDIFF_ROOT}/checkpoints/${FIRST_SEED_EXP}/latest.pth" ]; then
        if python -c "import sys, torch; c = torch.load(sys.argv[1], map_location='cpu', weights_only=False).get('config', {}); sys.exit(0 if c.get('use_stress_kpa', False) else 1)" "${POROUSDIFF_ROOT}/checkpoints/${FIRST_SEED_EXP}/latest.pth"; then
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
echo "Manifest: $MANIFEST | Base Experiment: $EXPERIMENT | Seeds: ${SEED_ARR[*]}"
nvidia-smi

EXP_LIST=()
for SEED_VAL in "${SEED_ARR[@]}"; do
    if [ "${#SEED_ARR[@]}" -gt 1 ]; then
        CUR_EXP="${EXPERIMENT}_s${SEED_VAL}"
    else
        CUR_EXP="$EXPERIMENT"
    fi
    EXP_LIST+=("$CUR_EXP")

    RESUME_ARGS=()
    if [ -f "${POROUSDIFF_ROOT}/checkpoints/${CUR_EXP}/latest.pth" ]; then
        echo "Existing checkpoint detected for ${CUR_EXP}; resuming automatically."
        RESUME_ARGS+=(--resume)
    fi

    python -u scripts/training/train_carbon_conditional_96.py \
        --experiment "$CUR_EXP" \
        --seed "$SEED_VAL" \
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

    HAS_OPENIMPALA=0
    if python -c "import openimpala" >/dev/null 2>&1; then
        HAS_OPENIMPALA=1
    fi

    REV_ARGS=()
    if [ "${RUN_REV_STUDY:-1}" = "1" ] && [ "$SEED_VAL" = "${SEED_ARR[0]}" ]; then
        REV_ARGS+=(--tiled-shape "${TILED_SHAPE:-96,192,192}")
        if [ "$HAS_OPENIMPALA" = "1" ]; then
            REV_ARGS+=(--run-rev-study)
        else
            echo "Warning: openimpala not installed in $VENV_DIR; generating tiled slab without --run-rev-study."
        fi
    fi

    python -u scripts/evaluation/sample_carbon_conditional_96.py \
        --experiment "$CUR_EXP" \
        --seed "$SEED_VAL" \
        --manifest-json "$MANIFEST" \
        --sampler "${SAMPLER:-ddim}" \
        --steps "${STEPS:-100}" \
        --guidance-scale "${GUIDANCE_SCALE:-1.5}" \
        --num-samples "${NUM_SAMPLES:-20}" \
        --batch-size "${BATCH_SIZE:-4}" \
        --sweep-loads "" \
        --export-openimpala \
        "${REV_ARGS[@]}"
done

EXP_CSV=$(IFS=,; echo "${EXP_LIST[*]}")
OUT_BENCH="${POROUSDIFF_ROOT}/results/${EXPERIMENT}/beamtime_benchmark"
OUT_CLAMP="${POROUSDIFF_ROOT}/results/${EXPERIMENT}/clamping_optimization"

VC_ARGS=(--include-virtual-compression)
if [ "${INCLUDE_LAGRANGIAN_SLAB:-1}" = "1" ]; then
    VC_ARGS+=(--include-lagrangian-slab --refine-strength "${REFINE_STRENGTH:-0.35}")
fi

OI_BENCH_ARGS=(--generate-openimpala-suite --export-real-openimpala --max-openimpala-per-method "${MAX_OPENIMPALA_PER_METHOD:-5}")
if [ "${RUN_OPENIMPALA:-1}" = "1" ]; then
    if python -c "import openimpala" >/dev/null 2>&1; then
        OI_BENCH_ARGS+=(--run-openimpala)
    else
        echo "Warning: openimpala not installed in $VENV_DIR; exported OpenImpala TIFF/INI suite without in-process PDE solve."
    fi
fi

python -u scripts/evaluation/evaluate_beamtime_multiplier.py \
    --experiments "porousdiff=${EXP_CSV}" \
    --data-root "$DATA_ROOT" \
    --out-dir "$OUT_BENCH" \
    --manifest-json "$MANIFEST" \
    --version "$VERSION" \
    --patch-size "$PATCH_SIZE" \
    "${VC_ARGS[@]}" \
    "${OI_BENCH_ARGS[@]}" \
    "${HELDOUT_ARGS[@]}"

OI_CLAMP_ARGS=()
if [ -f "${OUT_BENCH}/openimpala_transport_parity.csv" ]; then
    OI_CLAMP_ARGS+=(--openimpala-csv "${OUT_BENCH}/openimpala_transport_parity.csv")
fi

python -u scripts/evaluation/optimize_clamping_pressure.py \
    --experiments "porousdiff=${EXP_CSV}" \
    --data-root "$DATA_ROOT" \
    --out-dir "$OUT_CLAMP" \
    --manifest-json "$MANIFEST" \
    --version "$VERSION" \
    --patch-size "$PATCH_SIZE" \
    --dense-sweep-points "${DENSE_SWEEP_POINTS:-15}" \
    "${OI_CLAMP_ARGS[@]}"

if [ -n "${SLAB_SCAN_NPY:-}" ] && [ -f "$SLAB_SCAN_NPY" ]; then
    SLAB_RATIO_ARGS=()
    if [ -n "${SLAB_COMPRESSION_RATIOS:-}" ]; then
        SLAB_RATIO_ARGS+=(--target-compression-ratios "$SLAB_COMPRESSION_RATIOS")
    fi
    if [ -n "${SLAB_SOLID_FRACTIONS:-}" ]; then
        SLAB_RATIO_ARGS+=(--target-solid-fractions "$SLAB_SOLID_FRACTIONS")
    fi
    python -u scripts/evaluation/virtual_compress_scan.py \
        --input "$SLAB_SCAN_NPY" \
        --experiment "$FIRST_SEED_EXP" \
        --out-dir "${POROUSDIFF_ROOT}/results/${EXPERIMENT}/virtual_compression_lagrangian" \
        --mode lagrangian_slab \
        --source-load "${SLAB_SOURCE_LOAD:-50}" \
        --target-loads "${SLAB_TARGET_LOADS:-100,150,200,300}" \
        --material "${SLAB_MATERIAL:-virgin_carbon}" \
        --refine-strength "${REFINE_STRENGTH:-0.35}" \
        --export-openimpala \
        "${STRESS_ARGS[@]}" \
        "${SLAB_RATIO_ARGS[@]}"
fi

echo "Finished: $(date -u)"
