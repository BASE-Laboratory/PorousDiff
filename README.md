# PorousDiff

**Continuous compression-conditioned 3D diffusion modelling of porous electrode microstructures from synchrotron X-ray computed tomography (XCT).**

Developed by [BASE Laboratory](https://github.com/BASE-Laboratory) using NVIDIA GH200 Grace Hopper accelerators on Isambard-AI.

---

## Overview

PorousDiff trains a physics-validated 3D denoising diffusion surrogate across multi-resolution synchrotron XCT compression scans of porous battery electrodes—spanning **Virgin Carbon Felt**, **$\text{TiO}_2$-coated Carbon Felt Anodes**, and **Copper Hexacyanoferrate (CuHCF) Coated Cathodes** (plus **Diamond K11 DIAD** co-registered XCT/XRD scans).

Once trained, PorousDiff acts as a **synchrotron beamtime multiplier & virtual compression rig**:
1. **Continuous Mechanical Compression & Non-Affine Physics:** Given *any* compressive stress (`kPa`) or load (`mV/V`), generates 3D multi-phase microstructures (`0 = pore`, `1 = carbon fiber`, `2 = active TiO2/CuHCF coating`) at physical Representative Elementary Volume (REV) scale, capturing fiber buckling, `XY/Z` anisotropy, active-coating occlusion ($\text{SSA}_{\text{pore–coating}}$), triple-phase boundaries ($L_{\text{TPB}}$), electrical wiring, and geodesic tortuosity where heuristic numerical Z-squashing fails.
2. **Few-Shot Cross-Electrode Transfer:** Combines 1 densely scanned mechanical reference felt (Virgin Carbon Felt) with **only 2–3 sparse beamtime loads** of functionalized coated electrodes ($\text{TiO}_2$ or CuHCF) to predict intermediate coated-electrode compression states—cutting required synchrotron beamtime per new electrode chemistry by **3×–4×**.

---

### Key Capabilities (`v0.4` Unified Multi-Electrode & OpenImpala Optimization Pipeline)

1. **Calibrated 2-Phase & 3-Phase XCT Segmentation, Artifact Gating & REV Binning** (`porousdiff/preprocessing.py`, `scripts/preprocessing/prepare_local_drive_bundle.py`)
   - Masks exterior air outside the circular electrode punch (`--roi-radius-fraction 0.70`), subtracts pore-mode-clamped background, and applies shared per-`(material, beamline)` lowest-load thresholds (`delta_carbon`, `delta_coating`) separating pore (`0`), carbon scaffold (`1`), and high-$Z$ active deposits (`2`: $\text{TiO}_2$ / CuHCF), with a 10-second `--preview-only` Spyder diagnostic overlay (`qa/multiphase_overlay_diagnostic.png`).
   - Automatically excludes bullseye ring seams (`compute_radial_ring_score`) and compression plungers (`valid_slices.json`), and supports topology-preserving `--spatial-bin 2` REV pooling (`~260–380 µm` physical cubes).

2. **Physics-Informed Hierarchical Multi-Phase Diffusion (`v`-prediction + Zero-SNR + $\hat{x}_0$ Constraints)** (`porousdiff/models.py`, `porousdiff/losses.py`)
   - **Hierarchical 2-channel Zero-SNR representation (`c_solid`, `c_coating`):** Unifies 2-phase Virgin Felt and 3-phase $\text{TiO}_2$/CuHCF electrodes so Channel 0 transfers mechanical scaffold compression directly into sparse-load coated electrodes while Channel 1 learns active-coating morphology and triple-phase contacts.
   - **Physics-informed $\hat{x}_0$ training objective (`--physics-loss`):** Combines Min-SNR-$\gamma$ timestep weighting with 3D finite-difference interfacial sharpness, hierarchical no-floating-coating support penalties, and catalyst-to-scaffold mass-loading conservation, alongside automatic high-load delamination diagnostics (`coating_retention_diagnostics.json`).

3. **Zero-Beamtime Virtual *In-Situ* Compression & Seamless $>96^3$ OpenImpala REV Stitching** (`porousdiff/virtual_compression.py`, `scripts/evaluation/virtual_compress_scan.py`)
   - **Deterministic DDIM ODE Virtual Compression:** Inverts a real uncompressed (`0 kPa` / benchtop-CT) segmented scan along the probability-flow ODE and denoises it to arbitrary target stresses $\sigma$ without extra beamtime, enabling 1-to-1 comparison against `affine_z_squash` and `pillar_pore_deletion` on identical starting volumes (`--include-virtual-compression`).
   - **Seamless $>96^3$ 3D MultiDiffusion (`--tiled-shape`):** Synthesizes arbitrarily large stitch-free 3D electrode domains (e.g. `96 × 96 × 192` or `160³`) via 3D raised-cosine (Hann) overlapped denoising for OpenImpala REV convergence without extra training VRAM.

4. **Optimal Cell Clamping-Pressure ($\sigma^*$) Discovery + OpenImpala & DIAD XRD Coupling** (`porousdiff/optimization.py`, `scripts/evaluation/optimize_clamping_pressure.py`, `porousdiff/transport_and_xrd.py`)
   - Couples OpenImpala continuum transport (`openimpala_transport_parity.csv`) and 3D geodesic/Carman-Kozeny surrogates into a series-conductance Figure of Merit balancing ionic diffusivity ($D_{\text{eff},Z}/D_0$), electronic percolation ($\sigma_{\text{eff},Z}/\sigma_0$), and wired EASA/TPB—reporting $\sigma^*$ with bootstrap 95% confidence intervals across Real XCT, PorousDiff, and numerical baselines.
   - Quantifies 3D compressive pinch-points (`z_pinch_contact_density`, `coating_pinch_fraction`, `mean_coating_coordination`) directly correlated against co-registered DLS K11 DIAD XRD lattice microstrain/FWHM.

---

## Repository Structure

```text
PorousDiff/
├── porousdiff/
│   ├── __init__.py
│   ├── preprocessing.py         # Circular ROI, 2/3-phase segmentation, ring/plunger gating, 2x REV binning, kPa calibration
│   ├── baselines.py             # Literature heuristics: affine Z-rescaling & vertical pillar pore-deletion compression
│   ├── data.py                  # Hierarchical {0,1,2} <-> [-1,1]^2 encoding, balanced MultiElectrodePatchDataset, symmetries
│   ├── models.py                # Zero-SNR schedule, MultiModalConditionEmbedder (stress + material + voxel size), 3D UNet, EMA
│   ├── losses.py                # Physics-informed x0_hat loss (Min-SNR, 3D gradients, support, loading) & delamination check
│   ├── virtual_compression.py   # Deterministic DDIM ODE virtual scan compression & seamless >96^3 Hann MultiDiffusion
│   ├── optimization.py          # OpenImpala + 3D reaction-transport cell conductance FoM & bootstrap optimal sigma* finder
│   ├── metrics.py               # Porosity, EASA, TPB, wired coating, 3D pinch-points, chords, S2(r), tortuosity, OpenImpala
│   └── transport_and_xrd.py     # OpenImpala INI/batch suite generator & parser + DLS K11 DIAD XRD correlation
├── scripts/
│   ├── preprocessing/
│   │   ├── prepare_local_drive_bundle.py   # Spyder/CLI multi-electrode stream-bundler + 10s --preview-only overlay
│   │   ├── segment_carbon_xct.py
│   │   └── extract_carbon_3d_patches.py
│   ├── training/
│   │   └── train_carbon_conditional_96.py  # Multi-electrode 3-phase training + Few-Shot filtering + --physics-loss
│   └── evaluation/
│       ├── sample_carbon_conditional_96.py # Standard & seamless >96^3 (--tiled-shape) 3D sampling + OpenImpala TIFFs
│       ├── virtual_compress_scan.py        # Zero-beamtime DDIM ODE virtual compression of a real 0 kPa 3D scan
│       ├── evaluate_carbon_physics.py
│       ├── evaluate_beamtime_multiplier.py # Real XCT vs. Affine/Pillar vs. Virtual Compression vs. PorousDiff
│       ├── parse_openimpala_suite.py       # Ingest OpenImpala solver outputs into openimpala_transport_parity.csv
│       └── optimize_clamping_pressure.py   # Find optimal clamping pressure sigma* (95% CI) with OpenImpala + PorousDiff
├── slurm/                       # Isambard-AI GH200 Slurm batch scripts (including run_beamtime_multiplier.slurm)
├── tests/                       # Automated pytest unit + end-to-end pipeline verification
├── experiments.md               # Chronological experimental log
├── pyproject.toml
├── environment.yml
└── requirements-isambard.txt
```

---

## Quickstart & Handover Guide

### 1. Environment Setup (Isambard-AI GH200)

```bash
cd /projects/u6uc/asif/code/PorousDiff
git fetch origin && git checkout feature/paper-ready-conditional && git pull
source /projects/u6uc/asif/.venv/bin/activate
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
2. **Full Multi-Electrode REV Bundle** (recursively discovers nested `Virgin/`, `TiO2/Op001mVV`, `CuHCF/` folders, shares lowest-load thresholds per electrode, supports `--resume`, and writes a single `.zip` upload archive):
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

### 3. Training, OpenImpala Transport, Virtual Compression & Optimal Clamping $\sigma^*$ (Isambard-AI)

```bash
# Core Paper Run: Physics-informed multi-electrode training, sampling, virtual compression,
# OpenImpala batch suite generation, and preliminary optimal clamping-pressure sigma* sweep:
sbatch --export=ALL,EXPERIMENT="beamtime_fewshot_96",HELDOUT_STATES="tio2_0p018mVV tio2_0p033mVV cuhcf_0p018mVV cuhcf_0p033mVV" \
    slurm/run_beamtime_multiplier.slurm

# Run generated OpenImpala continuum diffusion/conduction jobs & re-solve exact optimal clamping sigma*:
bash results/beamtime_fewshot_96/beamtime_benchmark/openimpala_suite/run_openimpala_batch.sh
python scripts/evaluation/parse_openimpala_suite.py \
    --suite-dir results/beamtime_fewshot_96/beamtime_benchmark/openimpala_suite
python scripts/evaluation/optimize_clamping_pressure.py \
    --experiments porousdiff=beamtime_fewshot_96 \
    --data-root /projects/u6uc/asif/data/carbon_virgin \
    --manifest-json /projects/u6uc/asif/data/carbon_virgin/bundle_manifest.json

# Seamless >96^3 large-REV synthesis (e.g. 96x96x192 or 160^3) for OpenImpala REV convergence:
python scripts/evaluation/sample_carbon_conditional_96.py \
    --experiment beamtime_fewshot_96 \
    --manifest-json /projects/u6uc/asif/data/carbon_virgin/bundle_manifest.json \
    --tiled-shape 96,96,192 --num-samples 5 --export-openimpala

# Zero-beamtime virtual compression of a single uncompressed (0 kPa / Lab-CT) 3D scan:
python scripts/evaluation/virtual_compress_scan.py \
    --input /projects/u6uc/asif/data/lab_ct_0kpa.tif \
    --experiment beamtime_fewshot_96 \
    --source-load 0.0 --target-loads "50,150,300,500" \
    --material tio2_anode --export-openimpala
```

---

## Historical Benchmark Summary

Earlier unconditional baselines (`TiO₂`, `Op001mVV` / `Op026mVV`) established the motivation for the calibrated segmentation and Zero-SNR `v`-prediction formulation (full log in [`experiments.md`](experiments.md)):

| Experiment | Formulation | Patch Size | Real $\phi_s$ | Generated $\phi_s$ | Key Diagnostic Takeaway |
|---|---|---:|---:|---:|---|
| Exp 1–5 | 3D Conv Autoencoder (MSE + LPIPS) | `64³` | — | — | Reconstructs global envelope; smooths fine pores |
| Exp 6–11 | Grayscale 3D DDPM (`epsilon`, std linear) | `64³` | — | — | Blurry boundaries & horizontal banding |
| Exp 12 | Unconditional Binary DDPM (`epsilon`, std schedule) | `64³` | `0.1318` | `0.3948` | Non-zero terminal SNR shifts sparse binary mean ~3× toward `0.50` |
| Exp 13 | Unconditional Binary DDPM (`epsilon`, std schedule) | `96³` | `0.0984` | `0.2806` | `96³` feasible on GH200, but schedule/conditioning fixes required |
| **Exp 14 (`v0.3`)** | **Multi-Electrode 3-Phase Load-Conditioned DDPM (`v`-pred + Zero-SNR + REV)** | **`96³` (`2×` REV)** | **Virgin + $\text{TiO}_2$ + CuHCF + DIAD** | **Benchmarked vs. Z-squashing** | **Unifies 3-phase EASA/TPB prediction & 3×–4× synchrotron beamtime reduction** |

---

## Acknowledgements & License

Developed within [BASE Laboratory](https://github.com/BASE-Laboratory), with experiments performed on Isambard-AI NVIDIA GH200 computing resources (`brics.u6uc`). See [`LICENSE`](LICENSE) for licensing details.