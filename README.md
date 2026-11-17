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

## Key Capabilities (`v0.3` Unified Multi-Electrode Pipeline)

1. **Calibrated 2-Phase & 3-Phase XCT Segmentation, Artifact Gating & REV Binning** (`porousdiff/preprocessing.py`, `scripts/preprocessing/prepare_local_drive_bundle.py`)
   - Masks exterior air outside the circular electrode punch (`--roi-radius-fraction 0.70`), subtracts pore-mode-clamped background, and applies consistent physical thresholds (`delta_carbon`, `delta_coating`) separating pore (`0`), carbon scaffold (`1`), and high-$Z$ active deposits (`2`: $\text{TiO}_2$ / CuHCF), with a 10-second `--preview-only` Spyder diagnostic overlay (`qa/multiphase_overlay_diagnostic.png`).
   - Automatically excludes bullseye ring seams (`compute_radial_ring_score`) and compression plungers (`valid_slices.json`), and supports topology-preserving `--spatial-bin 2` REV pooling (`~260–380 µm` physical cubes).

2. **Hierarchical Multi-Phase + Multi-Modal Conditioning (`v`-prediction + Zero-SNR)** (`porousdiff/models.py`, `porousdiff/data.py`)
   - **Hierarchical 2-channel Zero-SNR representation (`c_solid`, `c_coating`):** Unifies 2-phase Virgin Felt and 3-phase $\text{TiO}_2$/CuHCF electrodes so Channel 0 transfers mechanical scaffold compression directly into sparse-load coated electrodes while Channel 1 learns active-coating morphology and triple-phase contacts.
   - **Multi-modal conditioning (`MultiModalConditionEmbedder`):** Conditions every UNet ResBlock on normalized **compressive stress / load**, **electrode chemistry ID** (`virgin_carbon`, `tio2_anode`, `cuhcf_cathode`), and **physical voxel size** ($\log(\Delta x / \Delta x_{\text{ref}})$ for cross-beamline DIAD compatibility), paired with Zero-Terminal-SNR schedules, `v_prediction`, EMA (`0.9995`), and $D_4(XY) \times \mathbb{Z}_2(Z)$ compression-safe symmetry.

3. **Head-to-Head Numerical Z-Compression Baselines & 3-Phase Electrochemical Suite** (`porousdiff/baselines.py`, `porousdiff/metrics.py`, `scripts/evaluation/evaluate_beamtime_multiplier.py`)
   - Directly benchmarks PorousDiff against standard literature heuristics (`affine_z_compress_to_target` and `pillar_pore_deletion_compress`) on real held-out synchrotron loads.
   - Evaluates **solid & coating fractions**, **EASA ($\text{SSA}_{\text{pore–coating}}$)**, **electronic contact area ($\text{SSA}_{\text{carbon–coating}}$)**, **triple-phase boundary density ($L_{\text{TPB}}$)**, **wired coating fraction**, **directional chord anisotropy**, **$S_2(r)$ RMSE**, **3D percolation**, **geodesic tortuosity**, and **OpenImpala** multi-page TIFF stacks.

---

## Repository Structure

```text
PorousDiff/
├── porousdiff/
│   ├── __init__.py
│   ├── preprocessing.py      # Circular ROI, 2/3-phase segmentation, ring/plunger gating, 2x REV binning, kPa calibration
│   ├── baselines.py          # Literature heuristics: affine Z-rescaling & vertical pillar pore-deletion compression
│   ├── data.py               # Hierarchical {0,1,2} <-> [-1,1]^2 encoding, balanced MultiElectrodePatchDataset, symmetries
│   ├── models.py             # Zero-SNR schedule, MultiModalConditionEmbedder (stress + material + voxel size), 3D UNet, EMA
│   └── metrics.py            # Porosity, EASA, TPB density, wired coating, chords, S2(r), percolation, tortuosity, OpenImpala
├── scripts/
│   ├── preprocessing/
│   │   ├── prepare_local_drive_bundle.py   # Spyder/CLI multi-electrode stream-bundler + 10s --preview-only overlay
│   │   ├── segment_carbon_xct.py
│   │   └── extract_carbon_3d_patches.py
│   ├── training/
│   │   └── train_carbon_conditional_96.py  # Single-material or multi-electrode 3-phase training + Few-Shot filtering
│   └── evaluation/
│       ├── sample_carbon_conditional_96.py
│       ├── evaluate_carbon_physics.py
│       └── evaluate_beamtime_multiplier.py # Real XCT vs. Affine/Pillar Z-Squash vs. PorousDiff benchmark
├── slurm/                    # Isambard-AI GH200 Slurm batch scripts (including run_beamtime_multiplier.slurm)
├── tests/                    # Automated pytest unit + end-to-end pipeline verification
├── experiments.md            # Chronological experimental log
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

### 3. Training, Sampling, OpenImpala Transport & DIAD XRD Evaluation (Isambard-AI)

```bash
# Core Paper Run: Unified Multi-Electrode Beamtime Multiplier (hold out intermediate TiO2 & CuHCF loads,
# sample all states, export OpenImpala .tif + INI/batch suite, and benchmark vs. Affine & Pillar Z-Squashing)
sbatch --export=ALL,EXPERIMENT="beamtime_fewshot_96",HELDOUT_STATES="tio2_0p018mVV tio2_0p033mVV cuhcf_0p018mVV cuhcf_0p033mVV" \
    slurm/run_beamtime_multiplier.slurm

# Run generated OpenImpala continuum diffusion/conduction jobs & aggregate parity plots:
bash results/beamtime_fewshot_96/beamtime_benchmark/openimpala_suite/run_openimpala_batch.sh
python scripts/evaluation/parse_openimpala_suite.py \
    --suite-dir results/beamtime_fewshot_96/beamtime_benchmark/openimpala_suite

# Optional: correlate co-registered DLS K11 DIAD XRD lattice strain / FWHM with 3D EASA & TPB:
python scripts/evaluation/evaluate_beamtime_multiplier.py \
    --experiments porousdiff=beamtime_fewshot_96 \
    --data-root /projects/u6uc/asif/data/beamtime_v03 \
    --manifest-json /projects/u6uc/asif/data/beamtime_v03/bundle_manifest.json \
    --include-repeats --beamline DLS_K11_DIAD \
    --xrd-csv /projects/u6uc/asif/data/diad_xrd_summary.csv
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