# PorousDiff

**Continuous compression-conditioned 3D diffusion modelling of porous electrode microstructures from synchrotron X-ray computed tomography (XCT).**

Developed by [BASE Laboratory](https://github.com/BASE-Laboratory) using NVIDIA GH200 Grace Hopper accelerators on Isambard-AI.

---

## Overview

PorousDiff provides a 3D denoising diffusion surrogate framework for multi-resolution synchrotron XCT compression scans of porous battery electrodes—spanning **Virgin Carbon Felt**, **$\text{TiO}_2$-coated Carbon Felt Anodes**, and **Copper Hexacyanoferrate (CuHCF) Coated Cathodes** (plus **Diamond K11 DIAD** co-registered XCT/XRD scans).

PorousDiff is designed to evaluate two core hypotheses on held-out synchrotron compression states:
1. **Continuous Mechanical Compression Conditioning:** Given a compressive stress (`kPa`) or load (`mV/V`) within the calibrated training range, generates 3D multi-phase microstructures (`0 = pore`, `1 = carbon fiber`, `2 = active TiO2/CuHCF coating`) to model fiber compaction, `XY/Z` chord anisotropy, active-coating occlusion ($\text{SSA}_{\text{pore–coating}}$), triple-phase boundaries ($L_{\text{TPB}}$), electrical wiring, and 26-connected Euclidean geodesic tortuosity relative to affine Z-squashing and vertical pillar pore-deletion baselines.
2. **Few-Shot Cross-Electrode Transfer Benchmark:** Combines a densely scanned mechanical reference felt (Virgin Carbon Felt) with a sparse subset (`2–3` loads) of functionalized coated-electrode scans ($\text{TiO}_2$ or CuHCF) to test how accurately intermediate held-out coated-electrode states can be interpolated compared with dense single-material training.

---

### Key Capabilities (`v0.5` Multi-Electrode & Native OpenImpala v4.6 Pipeline)

1. **Calibrated 2-Phase & 3-Phase XCT Segmentation, Partial-Volume Halo Cleanup & Mirror-Free Baseline Slabs** (`porousdiff/preprocessing.py`, `scripts/preprocessing/prepare_local_drive_bundle.py`)
   - Masks exterior air outside the circular electrode punch (`--roi-radius-fraction 0.70`), subtracts pore-mode-clamped background, applies shared per-`(material, beamline)` lowest-load thresholds (`delta_carbon`, `delta_coating`), and strips Paganin/PSF partial-volume "carbon rind" blur shells at `0–2` pore–coating boundaries (`--strip-halo-px 1`) while preserving $\ge 3\text{ px}$ carbon fibres and `0–1–2` triple-phase junctions.
   - Excludes bullseye ring seams (`compute_radial_ring_score`) and compression plungers (`valid_slices.json`), supports volume-fraction-matched `--spatial-bin 2` coarse-graining (with optional `audit_binning_topology_preservation` checks), extracts full-height `val/tall_patches.npy` columns so Z-compression baselines require zero mirror-tiling reflection, and falls back from Z-slab splitting to buffered XY-sector `train`/`val` splitting (`--split-mode auto`) on thin scans (`< 3 × raw_window`).

2. **Physics-Regularized Hierarchical Multi-Phase Diffusion (`v`-prediction + Zero-SNR + $\hat{x}_0$ Penalties)** (`porousdiff/models.py`, `porousdiff/losses.py`)
   - **Hierarchical 2-channel Zero-SNR representation (`c_solid`, `c_coating`) & Factored `load_only` CFG:** Aligns 2-phase Virgin Felt and 3-phase $\text{TiO}_2$/CuHCF electrodes via Channel 0 (`total solid`) and Channel 1 (`active coating`), conditioned on stress/load, categorical electrode chemistry (`nn.Embedding`), and `log(dx / dx_ref)` voxel size. Defaults multi-modal runs to `--cfg-mode load_only` (anchoring `material_id` and `voxel_size` during unconditional passes to prevent Virgin-to-Coated catalyst volume bleed) and enforces a single load-unit source of truth across training, `--resume`, sampling, and evaluation.
   - **Conditional-masked $\hat{x}_0$ auxiliary penalties (`--physics-loss`):** Combines Min-SNR-$\gamma$ timestep weighting with `cond_mask`-gated 3D finite-difference interfacial sharpness, universal hierarchical no-floating-coating support penalties, and uncoated-safe catalyst-to-scaffold mass-loading conservation, alongside high-load coating-retention diagnostics (`coating_retention_diagnostics.json`).

3. **Zero-Leakage Eulerian & Lagrangian Slab Virtual Compression + $>96^3$ MultiDiffusion REV** (`porousdiff/virtual_compression.py`, `scripts/evaluation/virtual_compress_scan.py`)
   - **Eulerian DDIM (`--mode eulerian_ddim`) & Zero-Leakage Lagrangian Slab (`--mode lagrangian_slab` / `--include-lagrangian-slab`):** Supports both fixed-box conditional DDIM ODE inversion (`invert_strength = 0.60`) and mass-conserving Lagrangian slab compression (`virtual_compress_lagrangian_slab`, affine Z-prewarping $Z_0 \to Z(\sigma)$ + `refine_strength = 0.35` 3D Hann MultiDiffusion refinement). On held-out synchrotron loads, `lagrangian_slab` never peeks at held-out XCT porosity (using PorousDiff blind $\hat{\phi}_s(\sigma)$, mechanical `strain_z`, or non-heldout PCHIP), while tagging `affine_z_squash` / `pillar_pore_deletion` as explicit oracle-porosity upper bounds.
   - **Seamless $>96^3$ 3D MultiDiffusion & OpenImpala REV (`--tiled-shape`, `--run-rev-study`):** Synthesizes larger 3D domains (e.g. `96 × 96 × 192` or `160³`) via 3D raised-cosine (Hann) overlapped denoising with automated OpenImpala sub-volume REV convergence and Z-profile uniformity audits.

4. **Per-Realization `OpenImpala` v4.6 PDE Coupling, Cauchy–Crofton Morphology, Constriction $\beta_Z(\sigma)$ & Static-Cell 1D Newman ASR ($\sigma^*$)** (`porousdiff/transport_and_xrd.py`, `porousdiff/metrics.py`, `porousdiff/optimization.py`)
   - **Per-realization in-process OpenImpala v4.6 Python API (`--run-openimpala`):** Solves steady-state PDE diffusivity/conductivity directly in Python (`tortuosity_all`, composite 3-phase `phase_diffusivities`, optional $3 \times 3$ `effective_diffusivity`, `rev_study`), pairing each OpenImpala solve 1-to-1 with its exact 3D realization (including `porousdiff_dense` sweeps) to preserve genuine within-load spatial bootstrap CIs on $\sigma^*$.
   - **Static Swagelok/coin/pouch Newman ASR & unbiased 3D stereology:** Defaults primary 1D Newman–Tobias ASR curves to static Swagelok/coin/pouch cells (`pump_penalty = 0`, thickness-scaled Nernst salt-depletion $R_{\text{mt}}(\sigma) = R_{\text{mt},0}\frac{L(\sigma)/L_0}{D_{\text{eff},Z}(\sigma)/D_0}$, pure $2/3$ Cauchy–Crofton `ssa_*_cc` kinetic area, with in-plane $K_{XY}$ pumping isolated as `flow_pumped_sensitivity`), alongside 3D cubical Euler density (`euler_density`), EDT pore radii, constriction $\beta_Z(\sigma)$, and $W_1$ / KS distances.

---

## Repository Structure

```text
PorousDiff/
├── porousdiff/
│   ├── __init__.py
│   ├── preprocessing.py         # Circular ROI, 2/3-phase segmentation, halo cleanup, 2x binning + topology audit, kPa calibration
│   ├── baselines.py             # Literature heuristics: affine Z-rescaling, pillar pore-deletion & mirror-tiling audit
│   ├── data.py                  # Hierarchical {0,1,2} <-> [-1,1]^2 encoding, load-unit source-of-truth helpers, symmetries
│   ├── models.py                # Zero-SNR schedule, MultiModalConditionEmbedder (load_only CFG), 3D UNet, EMA
│   ├── losses.py                # Conditional-masked x0_hat loss (Min-SNR, 3D gradients, support, uncoated-safe loading)
│   ├── virtual_compression.py   # Eulerian DDIM & Lagrangian slab virtual compression + >96^3 Hann MultiDiffusion
│   ├── optimization.py          # Static-cell Cauchy-Crofton 1D Newman ASR, per-patch OpenImpala CIs & constriction beta_Z
│   ├── metrics.py               # Cauchy-Crofton SSA, EASA, TPB, Euler chi_V, EDT pore radii, W1/KS & 26-conn tortuosity
│   └── transport_and_xrd.py     # Native OpenImpala v4.6 Python bridge, REV study, batch INI suite & K11 DIAD XRD analysis
├── scripts/
│   ├── preprocessing/
│   │   ├── prepare_local_drive_bundle.py   # Spyder/CLI multi-electrode stream-bundler + tall baseline slabs + --preview-only
│   │   ├── segment_carbon_xct.py
│   │   └── extract_carbon_3d_patches.py
│   ├── training/
│   │   └── train_carbon_conditional_96.py  # Multi-electrode 3-phase training + load_only CFG + --resume config guard
│   └── evaluation/
│       ├── audit_bundle_readiness.py       # Gate 0 pre-flight checker (units, patches, zero-overlap, phi_s monotonicity)
│       ├── sample_carbon_conditional_96.py # Standard & >96^3 (--tiled-shape, --run-rev-study) 3D sampling + OpenImpala TIFFs
│       ├── virtual_compress_scan.py        # Eulerian DDIM & Lagrangian slab virtual compression + XY fidelity audit
│       ├── evaluate_carbon_physics.py
│       ├── evaluate_beamtime_multiplier.py # Zero-leakage benchmark: Baselines vs. Lagrangian Slab vs. PorousDiff + OpenImpala
│       ├── parse_openimpala_suite.py       # Ingest OpenImpala solver outputs into openimpala_transport_parity.csv
│       └── optimize_clamping_pressure.py   # Static-cell Newman ASR & FoM sigma* + in-process dense-sweep OpenImpala PDE
├── legacy/scripts/              # Archived pre-conditional v0.1-v0.2 prototype scripts (Exps 1-8)
├── slurm/                       # Isambard-AI GH200 Slurm batch scripts (including run_v06_staged_campaign.slurm)
├── tests/                       # Automated pytest unit, OpenImpala v4.6 PDE, Red-Team regression, and CLI verification
├── experiments.md               # Chronological experimental log
├── pyproject.toml
├── environment.yml
└── requirements-isambard.txt
```

---

## Quickstart & Handover Guide

### 1. Environment Setup (Isambard-AI GH200 or Linux Workstation)

```bash
export POROUSDIFF_ROOT="/path/to/porousdiff_workspace"   # e.g. /projects/<project>/<user> on Isambard-AI
cd "$POROUSDIFF_ROOT/code/PorousDiff"
git fetch origin && git checkout feature/paper-ready-conditional && git pull
source "$POROUSDIFF_ROOT/.venv/bin/activate"
pip install --no-build-isolation --no-deps -e .
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 pytest -v
```

### 2. Local Drive Bundling (Windows Lab PC / Spyder)

1. **10-Second 3-Phase Threshold Preview** (verify `pore = black`, `carbon = cyan`, `TiO2/CuHCF = gold` on 1 scan per material first):
   ```bash
   python scripts/preprocessing/prepare_local_drive_bundle.py \
       --drive-root "D:/dry_samples" \
       --output-dir "D:/porousdiff_upload_bundle" \
       --datasets tio2_sample_folder cuhcf_sample_folder \
       --auto-multiphase --preview-only
   ```
2. **Full Multi-Electrode Bundle** (recursively discovers nested `Virgin/`, `TiO2/Op001mVV`, `CuHCF/` folders, shares lowest-load thresholds per electrode, saves mirror-free `val/tall_patches.npy` baseline columns, supports `--resume`, and writes a `.zip` upload archive):
   ```bash
   python scripts/preprocessing/prepare_local_drive_bundle.py \
       --drive-root "D:/dry_samples" \
       --output-dir "D:/porousdiff_upload_bundle" \
       --auto-multiphase --spatial-bin 2 \
       --voxel-size-um 1.6 --kpa-per-mvv 8500 \
       --beamline "DLS_I12" --resume --zip

   # Optional: merge a second drive/beamline (e.g. DLS K11 DIAD) into the SAME manifest without overwriting:
   python scripts/preprocessing/prepare_local_drive_bundle.py \
       --drive-root "E:/K11_DIAD_samples" \
       --output-dir "D:/porousdiff_upload_bundle" \
       --auto-multiphase --auto-calibrate-carbon \
       --voxel-size-um 0.54 --punch-diameter-mm 10.0 \
       --beamline "DLS_K11_DIAD" --append --resume --zip
   ```

### 3. Stage-Gated 3-Arm Beamtime Multiplier Campaign, `OpenImpala` v4.6 PDE & Static-Cell $\sigma^*$

```bash
# Recommended Staged Campaign (Gate 0 Audit -> Gate 1 1-Seed Virgin LOO -> Gate 2 3-Arm Sparse vs. Few-Shot vs. Oracle -> Gate 3 OpenImpala + Newman ASR):
sbatch --export=ALL,POROUSDIFF_ROOT="$POROUSDIFF_ROOT",STAGE="all",CAMPAIGN="v06_paper",HELDOUT_STATES="tio2_0p018mVV cuhcf_0p018mVV" \
    slurm/run_v06_staged_campaign.slurm

# Or run Gate 0 pre-flight readiness check standalone on login node (<15s CPU):
python scripts/evaluation/audit_bundle_readiness.py \
    --manifest-json "$POROUSDIFF_ROOT/data/multi_electrode/bundle_manifest.json" \
    --data-root "$POROUSDIFF_ROOT/data/multi_electrode"

# Execute OpenImpala v4.6 PDE transport directly in-process (--run-openimpala) & compute static-cell Newman ASR sigma*:
python scripts/evaluation/evaluate_beamtime_multiplier.py \
    --experiments porousdiff=beamtime_fewshot_96 \
    --data-root "$POROUSDIFF_ROOT/data/carbon_virgin" \
    --manifest-json "$POROUSDIFF_ROOT/data/carbon_virgin/bundle_manifest.json" \
    --include-lagrangian-slab --run-openimpala
python scripts/evaluation/optimize_clamping_pressure.py \
    --experiments porousdiff=beamtime_fewshot_96 \
    --data-root "$POROUSDIFF_ROOT/data/carbon_virgin" \
    --manifest-json "$POROUSDIFF_ROOT/data/carbon_virgin/bundle_manifest.json" \
    --run-openimpala

# MultiDiffusion >96^3 synthesis + automated OpenImpala REV sub-volume convergence study:
python scripts/evaluation/sample_carbon_conditional_96.py \
    --experiment beamtime_fewshot_96 \
    --manifest-json "$POROUSDIFF_ROOT/data/carbon_virgin/bundle_manifest.json" \
    --tiled-shape 96,96,192 --num-samples 5 --export-openimpala --run-rev-study

# Mass-conserving Lagrangian slab virtual compression of a tall low-load scan across target stresses:
python scripts/evaluation/virtual_compress_scan.py \
    --input "$POROUSDIFF_ROOT/data/low_load_68kpa_tall.tif" \
    --experiment beamtime_fewshot_96 \
    --mode lagrangian_slab --target-compression-ratios "0.85,0.72,0.60" \
    --source-load 68.0 --target-loads "150,300,450" \
    --material tio2_anode --export-openimpala
```

---

## Experimental & Pipeline Progression

Earlier unconditional baselines (`TiO₂`, `Op001mVV` / `Op026mVV`) established the motivation for the calibrated segmentation and Zero-SNR `v`-prediction formulation (full log in [`experiments.md`](experiments.md)):

| Stage | Formulation | Patch Size | Real $\phi_s$ | Generated $\phi_s$ | Status & Diagnostic Takeaway |
|---|---|---:|---:|---:|---|
| Exp 1–5 | 3D Conv Autoencoder (MSE + LPIPS) | `64³` | — | — | Completed — reconstructs global envelope; smooths fine pores |
| Exp 6–11 | Grayscale 3D DDPM (`epsilon`, std linear) | `64³` | — | — | Completed — blurry boundaries & horizontal banding |
| Exp 12 | Unconditional Binary DDPM (`epsilon`, std schedule) | `64³` | `0.1318` | `0.3948` | Completed — non-zero terminal SNR shifts sparse binary mean ~3× toward `0.50` |
| Exp 13 | Unconditional Binary DDPM (`epsilon`, std schedule) | `96³` | `0.0984` | `0.2806` | Completed — `96³` feasible on GH200, motivating Zero-SNR & conditioning fixes |
| Exp 14–16 (`v0.4`) | Multi-Electrode 3-Phase Load-Conditioned DDPM (`v`-pred + Zero-SNR + Newman ASR) | `96³` (`2×` binned) | Virgin + $\text{TiO}_2$ + CuHCF + DIAD | Verified on synthetic & unit suites | Unit/methodology safeguards, mirror-free slabs & 26-conn tortuosity |
| **Exp 17 (`v0.5`)** | **Native OpenImpala v4.6 PDE + Cauchy–Crofton ASR + Constriction $\beta_Z$ + Lagrangian Slab DDIM** | **`96³` & $>96^3$ REV** | **Virgin + $\text{TiO}_2$ + CuHCF + DIAD** | **Pending full GH200 campaign** | **In-process OpenImpala v4.6 PDE, Euler/EDT/$W_1$ metrics & Lagrangian slab verified in CI** |

---

## Acknowledgements & License

Developed within [BASE Laboratory](https://github.com/BASE-Laboratory), with experiments performed on Isambard-AI NVIDIA GH200 computing resources (`brics.u6uc`). See [`LICENSE`](LICENSE) for licensing details.