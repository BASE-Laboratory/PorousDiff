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

### Key Capabilities (`v0.4` Multi-Electrode & OpenImpala Pipeline)

1. **Calibrated 2-Phase & 3-Phase XCT Segmentation, Partial-Volume Halo Cleanup & Mirror-Free Baseline Slabs** (`porousdiff/preprocessing.py`, `scripts/preprocessing/prepare_local_drive_bundle.py`)
   - Masks exterior air outside the circular electrode punch (`--roi-radius-fraction 0.70`), subtracts pore-mode-clamped background, applies shared per-`(material, beamline)` lowest-load thresholds (`delta_carbon`, `delta_coating`), and strips Paganin/PSF partial-volume "carbon rind" blur shells at `0–2` pore–coating boundaries (`--strip-halo-px 1`) while preserving $\ge 3\text{ px}$ carbon fibres and `0–1–2` triple-phase junctions.
   - Excludes bullseye ring seams (`compute_radial_ring_score`) and compression plungers (`valid_slices.json`), supports volume-fraction-matched `--spatial-bin 2` coarse-graining (with optional `audit_binning_topology_preservation` checks), extracts full-height `val/tall_patches.npy` columns so Z-compression baselines require zero mirror-tiling reflection, and falls back from Z-slab splitting to buffered XY-sector `train`/`val` splitting (`--split-mode auto`) on thin scans (`< 3 × raw_window`).

2. **Physics-Regularized Hierarchical Multi-Phase Diffusion (`v`-prediction + Zero-SNR + $\hat{x}_0$ Penalties)** (`porousdiff/models.py`, `porousdiff/losses.py`)
   - **Hierarchical 2-channel Zero-SNR representation (`c_solid`, `c_coating`):** Aligns 2-phase Virgin Felt and 3-phase $\text{TiO}_2$/CuHCF electrodes via Channel 0 (`total solid`) and Channel 1 (`active coating`), conditioned on stress/load, categorical electrode chemistry (`nn.Embedding`), and `log(dx / dx_ref)` voxel size. Enforces a single load-unit source of truth across training, `--resume`, sampling, and evaluation so `mV/V` and `kPa` can never be mixed.
   - **Optional $\hat{x}_0$ auxiliary penalties (`--physics-loss`):** Combines Min-SNR-$\gamma$ timestep weighting with 3D finite-difference interfacial sharpness, hierarchical no-floating-coating support penalties, and catalyst-to-scaffold mass-loading conservation, alongside high-load coating-retention diagnostics (`coating_retention_diagnostics.json`).

3. **Conditional DDIM Microstructure Transfer & Seamless $>96^3$ MultiDiffusion Tiling** (`porousdiff/virtual_compression.py`, `scripts/evaluation/virtual_compress_scan.py`)
   - **Conditional DDIM Transfer (`invert_strength = 0.60`):** Partially inverts a real low-load segmented scan along the conditional DDIM trajectory and denoises it at higher in-range target stresses $\sigma$. Automatically audits `check_load_extrapolation`, roundtrip `same_load_solid_dice`, and `paired_xy_column_corr` vs. unpaired scans to verify specimen-specific structural retention rather than unconditional resampling (note: fixed Eulerian $96^3$ windows model mesoscale morphology rather than Lagrangian material-point tracking).
   - **Seamless $>96^3$ 3D MultiDiffusion (`--tiled-shape`):** Synthesizes larger 3D domains (e.g. `96 × 96 × 192` or `160³`) via symmetrically spaced 3D raised-cosine (Hann) overlapped denoising so domain-size REV convergence can be tested explicitly in OpenImpala.

4. **Optimal Clamping-Pressure ($\sigma^*$) Analysis via 1D Newman ASR ($\Omega\cdot\text{cm}^2$), Weight Sensitivity & DIAD XRD Partial Correlation** (`porousdiff/optimization.py`, `scripts/evaluation/optimize_clamping_pressure.py`, `porousdiff/transport_and_xrd.py`)
   - Couples uniform OpenImpala continuum transport (`openimpala_transport_parity.csv`, refusing mixed partial OpenImpala + surrogate curves) into both a normalized harmonic Figure of Merit (with an 11-point simplex `weight_sensitivity` sweep) and a **dimensional 1D Newman & Tobias porous-electrode Area-Specific Resistance $R_{\text{ASR}}(\sigma)$ ($\Omega\cdot\text{cm}^2$)** accounting for compressed thickness $L(\sigma) = L_0(1 - \epsilon_Z)$, Nernst mass-transfer resistance, clamping contact resistance, Darcy pumping penalty, and 3-regime sensitivity (`kinetics_limited`, `balanced`, `mass_transport_limited`).
   - Quantifies 3D compressive contact metrics (`specific_z_pinch_excess`, `coating_pinch_fraction`, `mean_coating_coordination`) correlated against co-registered DLS K11 DIAD XRD microstrain with OLS partial correlation (`partial_pearson_r_given_sf`) controlling for solid fraction $\phi_s$.

---

## Repository Structure

```text
PorousDiff/
├── porousdiff/
│   ├── __init__.py
│   ├── preprocessing.py         # Circular ROI, 2/3-phase segmentation, halo cleanup, 2x binning + topology audit, kPa calibration
│   ├── baselines.py             # Literature heuristics: affine Z-rescaling, pillar pore-deletion & mirror-tiling audit
│   ├── data.py                  # Hierarchical {0,1,2} <-> [-1,1]^2 encoding, load-unit source-of-truth helpers, symmetries
│   ├── models.py                # Zero-SNR schedule, MultiModalConditionEmbedder (stress + chemistry + voxel size), 3D UNet, EMA
│   ├── losses.py                # Auxiliary x0_hat loss (Min-SNR, 3D gradients, support, loading) & coating-retention check
│   ├── virtual_compression.py   # Conditional DDIM microstructure transfer, fidelity diagnostics & >96^3 Hann MultiDiffusion
│   ├── optimization.py          # 1D Newman ASR (Ohm*cm^2), simplex weight-sensitivity & bootstrap sigma* analysis
│   ├── metrics.py               # Porosity, EASA, TPB, wired coating, pinch excess, S2(r), 26-neighbour Euclidean tortuosity
│   └── transport_and_xrd.py     # OpenImpala INI/batch suite generator & parser + partial-correlation K11 DIAD XRD analysis
├── scripts/
│   ├── preprocessing/
│   │   ├── prepare_local_drive_bundle.py   # Spyder/CLI multi-electrode stream-bundler + tall baseline slabs + --preview-only
│   │   ├── segment_carbon_xct.py
│   │   └── extract_carbon_3d_patches.py
│   ├── training/
│   │   └── train_carbon_conditional_96.py  # Multi-electrode 3-phase training + --resume config guard + --physics-loss
│   └── evaluation/
│       ├── sample_carbon_conditional_96.py # Standard & >96^3 (--tiled-shape) 3D sampling + OpenImpala TIFFs
│       ├── virtual_compress_scan.py        # Conditional DDIM transfer across loads + same-load Dice & XY fidelity audit
│       ├── evaluate_carbon_physics.py
│       ├── evaluate_beamtime_multiplier.py # Real Held-Out XCT vs. Mirror-Free Baselines vs. PorousDiff
│       ├── parse_openimpala_suite.py       # Ingest OpenImpala solver outputs into openimpala_transport_parity.csv
│       └── optimize_clamping_pressure.py   # Newman ASR & FoM clamping-pressure sigma* + weight-sensitivity analysis
├── legacy/scripts/              # Archived pre-conditional v0.1-v0.2 prototype scripts (Exps 1-8)
├── slurm/                       # Isambard-AI GH200 Slurm batch scripts (configurable via POROUSDIFF_ROOT)
├── tests/                       # Automated pytest unit, trained-model DDIM fidelity, and end-to-end CLI verification
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
pytest -v
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

### 3. Training, OpenImpala Transport, Conditional DDIM Transfer & Clamping $\sigma^*$ Analysis

```bash
# Core Paper Benchmark: Multi-electrode training (auto-detecting calibrated stress_kpa),
# sampling, mirror-free baseline benchmark, OpenImpala suite generation, and preliminary sigma* sweep:
sbatch --export=ALL,POROUSDIFF_ROOT="$POROUSDIFF_ROOT",EXPERIMENT="beamtime_fewshot_96",HELDOUT_STATES="tio2_0p018mVV tio2_0p033mVV cuhcf_0p018mVV cuhcf_0p033mVV" \
    slurm/run_beamtime_multiplier.slurm

# Run generated OpenImpala continuum diffusion/conduction jobs & re-evaluate Newman ASR + sigma*:
bash "$POROUSDIFF_ROOT/results/beamtime_fewshot_96/beamtime_benchmark/openimpala_suite/run_openimpala_batch.sh"
python scripts/evaluation/parse_openimpala_suite.py \
    --suite-dir "$POROUSDIFF_ROOT/results/beamtime_fewshot_96/beamtime_benchmark/openimpala_suite"
python scripts/evaluation/optimize_clamping_pressure.py \
    --experiments porousdiff=beamtime_fewshot_96 \
    --data-root "$POROUSDIFF_ROOT/data/carbon_virgin" \
    --manifest-json "$POROUSDIFF_ROOT/data/carbon_virgin/bundle_manifest.json"

# MultiDiffusion >96^3 synthesis (e.g. 96x96x192 or 160^3) to test OpenImpala REV domain-size convergence:
python scripts/evaluation/sample_carbon_conditional_96.py \
    --experiment beamtime_fewshot_96 \
    --manifest-json "$POROUSDIFF_ROOT/data/carbon_virgin/bundle_manifest.json" \
    --tiled-shape 96,96,192 --num-samples 5 --export-openimpala

# Conditional DDIM transfer of a real low-load (in-range, e.g. 68 kPa) 3D scan to higher target stresses:
python scripts/evaluation/virtual_compress_scan.py \
    --input "$POROUSDIFF_ROOT/data/low_load_68kpa.tif" \
    --experiment beamtime_fewshot_96 \
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
| **Exp 14–16 (`v0.4`)** | **Multi-Electrode 3-Phase Load-Conditioned DDPM (`v`-pred + Zero-SNR + OpenImpala ASR)** | **`96³` (`2×` binned)** | **Virgin + $\text{TiO}_2$ + CuHCF + DIAD** | **Pending full GH200 benchmark** | **Software pipeline & unit/methodology guards verified; ready for held-out synchrotron evaluation** |

---

## Acknowledgements & License

Developed within [BASE Laboratory](https://github.com/BASE-Laboratory), with experiments performed on Isambard-AI NVIDIA GH200 computing resources (`brics.u6uc`). See [`LICENSE`](LICENSE) for licensing details.