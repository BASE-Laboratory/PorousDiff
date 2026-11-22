#!/bin/bash
#SBATCH --job-name=porousdiff-v06-staged
#SBATCH --account=brics.u6uc
#SBATCH --partition=workq
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --output=/projects/u6uc/asif/logs/porousdiff-v06-%j.out
#SBATCH --error=/projects/u6uc/asif/logs/porousdiff-v06-%j.err

# Stage-gated Red-Team Campaign Runner (STAGE = gate0 | gate1 | gate2 | gate3 | all)
#   gate0: Fast CPU pre-flight audit (units, patches, zero-overlap, phi_s monotonicity)
#   gate1: 1-seed Virgin Carbon Felt Leave-One-Out sanity gate (hard Go/No-Go assertions)
#   gate2: Controlled 3-Arm Beamtime Multiplier Ablation on identical held-out coated states:
#            Arm A (sparse_no_transfer): K=2 endpoint coated scans ONLY (no Virgin Felt)
#            Arm B (fewshot_multiplier): Dense Virgin Carbon Felt + same K=2 endpoint coated scans
#            Arm C (full_oracle):        All non-heldout coated + Virgin scans
#   gate3: Unified zero-leakage benchmark (Lagrangian slab + OpenImpala PDE + static Newman ASR sigma*)

set -euo pipefail

export POROUSDIFF_ROOT="${POROUSDIFF_ROOT:-/projects/u6uc/asif}"
CODE_DIR="${CODE_DIR:-${POROUSDIFF_ROOT}/code/PorousDiff}"
VENV_DIR="${VENV_DIR:-${POROUSDIFF_ROOT}/.venv}"
DATA_ROOT="${DATA_ROOT:-${POROUSDIFF_ROOT}/data/multi_electrode}"
MANIFEST="${MANIFEST:-${DATA_ROOT}/bundle_manifest.json}"

cd "$CODE_DIR"
source "$VENV_DIR/bin/activate"
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"

STAGE="${STAGE:-all}"
CAMPAIGN="${CAMPAIGN:-v06_paper}"
VERSION="${VERSION:-carbon_calibrated_v1}"
PATCH_SIZE="${PATCH_SIZE:-96}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
LR="${LR:-2e-4}"
STEPS="${STEPS:-100}"
NUM_SAMPLES="${NUM_SAMPLES:-20}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.5}"
SEEDS="${SEEDS:-42 43 44}"
read -r -a SEED_ARR <<< "$SEEDS"

HAS_OPENIMPALA=0
if python -c "import openimpala" >/dev/null 2>&1; then
    HAS_OPENIMPALA=1
fi

STRESS_ARGS=()
if [ "${USE_STRESS_KPA:-auto}" = "1" ]; then
    STRESS_ARGS+=(--use-stress-kpa)
elif [ "${USE_STRESS_KPA:-auto}" = "auto" ] && [ -f "$MANIFEST" ]; then
    if python -c "import json, sys; s = json.load(open(sys.argv[1])).get('scans', []); sys.exit(0 if (s and all(x.get('stress_kpa') is not None for x in s)) else 1)" "$MANIFEST"; then
        STRESS_ARGS+=(--use-stress-kpa)
    fi
fi

PHYSICS_ARGS=()
if [ "${PHYSICS_LOSS:-1}" = "1" ]; then
    PHYSICS_ARGS+=(--physics-loss)
fi

run_gate0() {
    echo "=== [Gate 0] Pre-Flight Bundle Readiness Audit ==="
    AUDIT_JSON="${POROUSDIFF_ROOT}/results/${CAMPAIGN}/gate0_bundle_audit.json"
    python -u scripts/evaluation/audit_bundle_readiness.py \
        --manifest-json "$MANIFEST" \
        --data-root "$DATA_ROOT" \
        --version "$VERSION" \
        --patch-size "$PATCH_SIZE" \
        --monotonic-slack "${MONOTONIC_SLACK:-0.015}" \
        --out-json "$AUDIT_JSON"
}

run_gate1() {
    echo "=== [Gate 1] 1-Seed Virgin Carbon Felt LOO Sanity Gate ==="
    # Automatically select virgin_carbon states and pick middle load as GATE1_HELDOUT if unset
    VIRGIN_INFO=$(python -c "
import json, sys
m = json.load(open(sys.argv[1]))
v = [s for s in m.get('scans', []) if not s.get('is_repeat_or_secondary', False) and s.get('material', 'virgin_carbon') == 'virgin_carbon']
v.sort(key=lambda x: float(x['stress_kpa'] if x.get('stress_kpa') is not None else x.get('load_mv_per_v', 0.0)))
if len(v) < 3:
    sys.exit('Need >= 3 virgin_carbon scans in manifest for Gate 1 LOO.')
mid = v[len(v) // 2]['state']
all_st = ' '.join(x['state'] for x in v)
train_st = ' '.join(x['state'] for x in v if x['state'] != mid)
print(f'{mid}|{all_st}|{train_st}')
" "$MANIFEST")

    AUTO_MID="${VIRGIN_INFO%%|*}"
    REST="${VIRGIN_INFO#*|}"
    VIRGIN_ALL="${REST%%|*}"
    AUTO_TRAIN="${REST#*|}"

    G1_HELDOUT="${GATE1_HELDOUT:-$AUTO_MID}"
    read -r -a VIRGIN_ALL_ARR <<< "$VIRGIN_ALL"
    G1_TRAIN_ARR=()
    for st in "${VIRGIN_ALL_ARR[@]}"; do
        if [ "$st" != "$G1_HELDOUT" ]; then
            G1_TRAIN_ARR+=("$st")
        fi
    done

    G1_EXP="${CAMPAIGN}_gate1_virgin_loo"
    G1_SEED="${SEED_ARR[0]}"
    G1_RESUME=()
    if [ -f "${POROUSDIFF_ROOT}/checkpoints/${G1_EXP}/latest.pth" ]; then
        G1_RESUME+=(--resume)
    fi

    python -u scripts/training/train_carbon_conditional_96.py \
        --experiment "$G1_EXP" \
        --seed "$G1_SEED" \
        --data-root "$DATA_ROOT" \
        --manifest-json "$MANIFEST" \
        --version "$VERSION" \
        --patch-size "$PATCH_SIZE" \
        --include-states "${G1_TRAIN_ARR[@]}" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --grad-accum "$GRAD_ACCUM" \
        --lr "$LR" \
        --amp bf16 \
        --num-workers 4 \
        "${STRESS_ARGS[@]}" \
        "${PHYSICS_ARGS[@]}" \
        "${G1_RESUME[@]}"

    python -u scripts/evaluation/sample_carbon_conditional_96.py \
        --experiment "$G1_EXP" \
        --seed "$G1_SEED" \
        --manifest-json "$MANIFEST" \
        --eval-states "${VIRGIN_ALL_ARR[@]}" \
        --sampler ddim \
        --steps "$STEPS" \
        --guidance-scale "$GUIDANCE_SCALE" \
        --num-samples "$NUM_SAMPLES" \
        --batch-size "$BATCH_SIZE" \
        --sweep-loads "" \
        --export-openimpala

    G1_BENCH="${POROUSDIFF_ROOT}/results/${G1_EXP}/beamtime_benchmark"
    OI_FLAG=()
    if [ "$HAS_OPENIMPALA" = "1" ] && [ "${RUN_OPENIMPALA:-1}" = "1" ]; then
        OI_FLAG+=(--run-openimpala --max-openimpala-per-method "${MAX_OPENIMPALA_PER_METHOD:-5}")
    fi

    python -u scripts/evaluation/evaluate_beamtime_multiplier.py \
        --experiments "porousdiff=${G1_EXP}" \
        --data-root "$DATA_ROOT" \
        --out-dir "$G1_BENCH" \
        --manifest-json "$MANIFEST" \
        --version "$VERSION" \
        --patch-size "$PATCH_SIZE" \
        --heldout-states "$G1_HELDOUT" \
        "${OI_FLAG[@]}"

    # Hard Go/No-Go Gate 1 assertion on held-out Virgin load
    python -c "
import json, sys
rep = json.load(open(sys.argv[1]))
st = sys.argv[2]
m = rep[st]['methods']
pd_s2 = float(m['porousdiff']['s2_rmse'])
noop_s2 = float(m['noop_uncompressed']['s2_rmse'])
real_sf = float(rep[st]['real']['metrics']['solid_fraction']['mean'])
pred_sf = float(m['porousdiff']['metrics']['solid_fraction']['mean'])
sf_err = abs(pred_sf - real_sf)
print(f'[Gate 1 Check | {st}] porousdiff s2_rmse={pd_s2:.5f} vs noop={noop_s2:.5f} | abs_err_phi_s={sf_err:.4f}')
if pd_s2 >= noop_s2:
    sys.exit(f'Gate 1 FAILED: porousdiff s2_rmse ({pd_s2:.5f}) did not beat noop_uncompressed ({noop_s2:.5f}) on {st}.')
if sf_err > float(sys.argv[3]):
    sys.exit(f'Gate 1 FAILED: abs_err_solid_fraction ({sf_err:.4f}) exceeded threshold ({sys.argv[3]}) on {st}.')
print('Gate 1 PASSED.')
" "${G1_BENCH}/beamtime_multiplier_report.json" "$G1_HELDOUT" "${GATE1_MAX_SF_ERR:-0.035}"
}

train_and_sample_arm() {
    local arm_base="$1"
    shift
    local include_states=("$@")

    for SEED_VAL in "${SEED_ARR[@]}"; do
        if [ "${#SEED_ARR[@]}" -gt 1 ]; then
            CUR_EXP="${arm_base}_s${SEED_VAL}"
        else
            CUR_EXP="$arm_base"
        fi
        RESUME_ARGS=()
        if [ -f "${POROUSDIFF_ROOT}/checkpoints/${CUR_EXP}/latest.pth" ]; then
            RESUME_ARGS+=(--resume)
        fi

        python -u scripts/training/train_carbon_conditional_96.py \
            --experiment "$CUR_EXP" \
            --seed "$SEED_VAL" \
            --data-root "$DATA_ROOT" \
            --manifest-json "$MANIFEST" \
            --version "$VERSION" \
            --patch-size "$PATCH_SIZE" \
            --include-states "${include_states[@]}" \
            --balanced-sampling \
            --epochs "$EPOCHS" \
            --batch-size "$BATCH_SIZE" \
            --grad-accum "$GRAD_ACCUM" \
            --lr "$LR" \
            --amp bf16 \
            --num-workers 4 \
            "${STRESS_ARGS[@]}" \
            "${PHYSICS_ARGS[@]}" \
            "${RESUME_ARGS[@]}"

        REV_ARGS=()
        if [ "${RUN_REV_STUDY:-1}" = "1" ] && [ "$SEED_VAL" = "${SEED_ARR[0]}" ] && [[ "$arm_base" == *"fewshot"* ]]; then
            REV_ARGS+=(--tiled-shape "${TILED_SHAPE:-96,192,192}")
            if [ "$HAS_OPENIMPALA" = "1" ]; then
                REV_ARGS+=(--run-rev-study)
            fi
        fi

        python -u scripts/evaluation/sample_carbon_conditional_96.py \
            --experiment "$CUR_EXP" \
            --seed "$SEED_VAL" \
            --manifest-json "$MANIFEST" \
            --sampler ddim \
            --steps "$STEPS" \
            --guidance-scale "$GUIDANCE_SCALE" \
            --num-samples "$NUM_SAMPLES" \
            --batch-size "$BATCH_SIZE" \
            --sweep-loads "" \
            --export-openimpala \
            "${REV_ARGS[@]}"
    done
}

resolve_three_arms() {
    # Resolves SPARSE_COATED_STATES, FEWSHOT_STATES, ORACLE_STATES, and HELDOUT_COATED_STATES from MANIFEST
    python -c "
import json, os, sys
m = json.load(open(sys.argv[1]))
scans = [s for s in m.get('scans', []) if not s.get('is_repeat_or_secondary', False)]
by_mat = {}
for s in scans:
    by_mat.setdefault(s.get('material', 'virgin_carbon'), []).append(s)
for mat_s in by_mat.values():
    mat_s.sort(key=lambda x: float(x['stress_kpa'] if x.get('stress_kpa') is not None else x.get('load_mv_per_v', 0.0)))

virgin = [x['state'] for x in by_mat.get('virgin_carbon', [])]
explicit_held = set(os.environ.get('HELDOUT_STATES', '').split())
explicit_sparse = os.environ.get('SPARSE_COATED_ENDPOINTS', '').split()

sparse_coated = []
heldout_coated = []
oracle_coated = []
for mat, items in sorted(by_mat.items()):
    if mat == 'virgin_carbon':
        continue
    if len(items) < 3:
        sys.exit(f'Coated material {mat} needs >= 3 states (2 endpoints + >= 1 held-out interior), found {len(items)}.')
    if explicit_held:
        h_m = [x['state'] for x in items if x['state'] in explicit_held]
    else:
        # Hold out interior scan(s), keep lowest & highest endpoints for K=2 sparse/few-shot
        h_m = [x['state'] for x in items[1:-1]]
    heldout_coated.extend(h_m)
    non_held = [x['state'] for x in items if x['state'] not in h_m]
    oracle_coated.extend(non_held)
    if explicit_sparse:
        sparse_coated.extend([x for x in non_held if x in explicit_sparse])
    else:
        sparse_coated.extend([items[0]['state'], items[-1]['state']])

if not sparse_coated or not heldout_coated:
    sys.exit('Could not resolve coated sparse endpoints and held-out interior states from manifest.')
print('SPARSE=' + ' '.join(sparse_coated))
print('FEWSHOT=' + ' '.join(virgin + sparse_coated))
print('ORACLE=' + ' '.join(virgin + oracle_coated))
print('HELDOUT=' + ' '.join(heldout_coated))
" "$MANIFEST"
}

run_gate2() {
    echo "=== [Gate 2] Controlled 3-Arm Beamtime Multiplier Ablation ==="
    ARM_CFG=$(resolve_three_arms)
    SPARSE_STR=$(echo "$ARM_CFG" | awk -F= '/^SPARSE=/{print $2}')
    FEWSHOT_STR=$(echo "$ARM_CFG" | awk -F= '/^FEWSHOT=/{print $2}')
    ORACLE_STR=$(echo "$ARM_CFG" | awk -F= '/^ORACLE=/{print $2}')
    HELDOUT_STR=$(echo "$ARM_CFG" | awk -F= '/^HELDOUT=/{print $2}')

    echo "  Held-out coated states : $HELDOUT_STR"
    echo "  Arm A (Sparse Coated)  : $SPARSE_STR"
    echo "  Arm B (Few-Shot + Vir) : $FEWSHOT_STR"
    echo "  Arm C (Full Oracle)    : $ORACLE_STR"

    read -r -a SPARSE_ARR <<< "$SPARSE_STR"
    read -r -a FEWSHOT_ARR <<< "$FEWSHOT_STR"
    read -r -a ORACLE_ARR <<< "$ORACLE_STR"

    train_and_sample_arm "${CAMPAIGN}_armA_sparse" "${SPARSE_ARR[@]}"
    train_and_sample_arm "${CAMPAIGN}_armB_fewshot" "${FEWSHOT_ARR[@]}"
    train_and_sample_arm "${CAMPAIGN}_armC_oracle" "${ORACLE_ARR[@]}"
}

arm_csv_for() {
    local arm_base="$1"
    if [ "${#SEED_ARR[@]}" -gt 1 ]; then
        local parts=()
        for sv in "${SEED_ARR[@]}"; do
            parts+=("${arm_base}_s${sv}")
        done
        (IFS=,; echo "${parts[*]}")
    else
        echo "$arm_base"
    fi
}

run_gate3() {
    echo "=== [Gate 3] Unified Zero-Leakage Evaluation & Static-Cell Newman ASR sigma* ==="
    ARM_CFG=$(resolve_three_arms)
    HELDOUT_STR=$(echo "$ARM_CFG" | awk -F= '/^HELDOUT=/{print $2}')
    read -r -a HELDOUT_ARR <<< "$HELDOUT_STR"

    SPARSE_CSV=$(arm_csv_for "${CAMPAIGN}_armA_sparse")
    FEWSHOT_CSV=$(arm_csv_for "${CAMPAIGN}_armB_fewshot")
    ORACLE_CSV=$(arm_csv_for "${CAMPAIGN}_armC_oracle")

    OUT_BENCH="${POROUSDIFF_ROOT}/results/${CAMPAIGN}/beamtime_benchmark"
    OUT_CLAMP="${POROUSDIFF_ROOT}/results/${CAMPAIGN}/clamping_optimization"

    OI_BENCH_ARGS=(--generate-openimpala-suite --export-real-openimpala --max-openimpala-per-method "${MAX_OPENIMPALA_PER_METHOD:-5}")
    OI_CLAMP_ARGS=()
    if [ "$HAS_OPENIMPALA" = "1" ] && [ "${RUN_OPENIMPALA:-1}" = "1" ]; then
        OI_BENCH_ARGS+=(--run-openimpala)
        OI_CLAMP_ARGS+=(--run-openimpala)
    fi

    # Put fewshot_multiplier first so virtual compression & Lagrangian slab use the Beamtime Multiplier checkpoint
    python -u scripts/evaluation/evaluate_beamtime_multiplier.py \
        --experiments \
            "fewshot_multiplier=${FEWSHOT_CSV}" \
            "sparse_no_transfer=${SPARSE_CSV}" \
            "full_oracle=${ORACLE_CSV}" \
        --data-root "$DATA_ROOT" \
        --out-dir "$OUT_BENCH" \
        --manifest-json "$MANIFEST" \
        --version "$VERSION" \
        --patch-size "$PATCH_SIZE" \
        --include-virtual-compression \
        --include-lagrangian-slab \
        --refine-strength "${REFINE_STRENGTH:-0.35}" \
        --heldout-states "${HELDOUT_ARR[@]}" \
        "${OI_BENCH_ARGS[@]}"

    if [ -f "${OUT_BENCH}/openimpala_transport_parity.csv" ]; then
        OI_CLAMP_ARGS+=(--openimpala-csv "${OUT_BENCH}/openimpala_transport_parity.csv")
    fi

    python -u scripts/evaluation/optimize_clamping_pressure.py \
        --experiments \
            "fewshot_multiplier=${FEWSHOT_CSV}" \
            "sparse_no_transfer=${SPARSE_CSV}" \
            "full_oracle=${ORACLE_CSV}" \
        --data-root "$DATA_ROOT" \
        --out-dir "$OUT_CLAMP" \
        --manifest-json "$MANIFEST" \
        --version "$VERSION" \
        --patch-size "$PATCH_SIZE" \
        --dense-sweep-points "${DENSE_SWEEP_POINTS:-15}" \
        "${OI_CLAMP_ARGS[@]}"
}

echo "Started Staged Campaign ($STAGE): $(date -u)"
case "$STAGE" in
    gate0)
        run_gate0
        ;;
    gate1)
        run_gate0
        nvidia-smi
        run_gate1
        ;;
    gate2)
        run_gate0
        nvidia-smi
        run_gate2
        ;;
    gate3)
        nvidia-smi
        run_gate3
        ;;
    all)
        run_gate0
        nvidia-smi
        run_gate1
        run_gate2
        run_gate3
        ;;
    *)
        echo "Unknown STAGE=${STAGE} (expected gate0|gate1|gate2|gate3|all)" >&2
        exit 2
        ;;
esac
echo "Finished Staged Campaign ($STAGE): $(date -u)"
