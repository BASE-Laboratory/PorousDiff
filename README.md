# PorousDiff

**Continuous compression-conditioned 3D diffusion modelling of porous electrode microstructures from synchrotron X-ray computed tomography (XCT).**

Developed by [BASE Laboratory](https://github.com/BASE-Laboratory) using NVIDIA GH200 Grace Hopper accelerators on Isambard-AI.

---

## Overview

PorousDiff trains a physics-validated 3D denoising diffusion model across synchrotron XCT scans of carbon-felt electrodes acquired under progressive mechanical compression (`0.008` to `0.053 mV/V` load-cell states). Once trained, the model acts as a **virtual compression rig**: given *any* continuous scalar load value—including unseen intermediate compressions—it generates 3D binary porous microstructures (`1 = solid fiber`, `0 = pore`) matching experimental solid fraction, specific surface area, through-plane (`Z`) vs. in-plane (`XY`) chord-length anisotropy, two-point spatial correlation $S_2(r)$, 3D solid/pore percolation, and geodesic pore tortuosity. Generated volumes export directly as multi-page `uint8` TIFF stacks for downstream transport simulations in OpenImpala.

---

## Key Capabilities (`v0.2` Conditional Pipeline)

1. **Calibrated XCT Segmentation & Automatic Artifact Gating** (`porousdiff/preprocessing.py`, `scripts/preprocessing/segment_carbon_xct.py`)
   - Masks out exterior air surrounding the circular electrode punch (`--roi-radius-fraction 0.70`).
   - Subtracts a pore-mode-clamped low-frequency background and applies a single physical `delta_threshold` across all compression states, preventing sparse low-load slices (`0.008 mV/V`) from suffering per-slice Otsu inversion while preserving fiber densification under heavy compression.
   - Automatically detects and excludes non-physical mid-stack bullseye ring/stitch seams (via azimuthal radial variance scoring) and the rising bottom compression plunger (`valid_slices.json`).

2. **Z-Stratified, Leak-Free 3D Patch Extraction** (`scripts/preprocessing/extract_carbon_3d_patches.py`)
   - Enforces a spatial buffer of at least `patch_size` slices between `train` and `val` Z-slabs so zero voxels or adjacent boundary fibers leak across splits.
   - Uses round-robin Z-window stratification so `max_patches` spans the entire clean thickness of the felt rather than saturating inside the top few Z windows.
   - Writes stacked `patches.npy` archives for fast single-read RAM caching on HPC lustre filesystems.

3. **Continuous Scalar Load-Conditioned 3D Diffusion (`v`-prediction + Zero-SNR)** (`porousdiff/models.py`, `scripts/training/train_carbon_conditional_96.py`)
   - **Multi-scale conditioning (`ContinuousLoadConditionedUNet`):** Maps normalized continuous load into a smooth Fourier/sinusoidal embedding injected into `temb` across every encoder, bottleneck, and decoder ResBlock (`96³`, `48³`, `24³`), paired with 3-channel input concatenation `[x_t, cond_norm * mask, mask]` (`in_channels = 3`). Eliminates the 1-token cross-attention softmax degeneracy (`softmax([s]) == 1`) and prevents midpoint (`0.0`) vs. unconditional CFG (`mask = 0`) collisions.
   - **Unbiased porosity (`zero_snr_scaled_linear` + `v_prediction` + EMA):** Rescales the noise schedule to exact zero terminal SNR ($\bar{\alpha}_{T-1} = 0$) with velocity (`v`) prediction and EMA (`0.9995`) weights, eliminating residual signal leakage that previously biased generated solid fractions ~3× upward toward `0.50`.
   - **Anisotropy-preserving 3D augmentation:** Samples from the 16-element symmetry group $D_4(XY) \times \mathbb{Z}_2(Z)$ that rotates/flips in-plane `XY` and flips `Z` without ever swapping the vertical compression axis `Z` with `X` or `Y`.
   - **GH200 `bf16` throughput:** `torch.amp.autocast('cuda', dtype=torch.bfloat16)` with `batch_size=4, grad_accum=2` completes 100 epochs on `96³` patches inside a single 6-hour GH200 Slurm allocation, with full RNG/EMA/optimizer `--resume` support.

4. **Unseen Load Interpolation & 3D Physics Validation** (`scripts/evaluation/sample_carbon_conditional_96.py`, `scripts/evaluation/evaluate_carbon_physics.py`)
   - Supports fast 100-step DDIM (`set_alpha_to_one=False`) and 1000-step DDPM sampling with classifier-free guidance (`--guidance-scale 1.5`).
   - Generates fixed-seed continuous compression sweeps (`--sweep-loads`) showing the same fiber network progressively compressing along `Z`.
   - Evaluates and plots real vs. generated **solid fraction**, **specific surface area (SSA)**, **directional pore/solid chord lengths & `XY/Z` anisotropy**, **directional $S_2(r)$**, **6-connected solid integrity & pore percolation**, and **3D geodesic pore tortuosity** (`--heldout-states` highlights unseen interpolated loads).

---

## Repository Structure

```text
PorousDiff/
├── porousdiff/
│   ├── __init__.py
│   ├── preprocessing.py      # Circular ROI, pore-referenced segmentation, ring/plunger gating, Z-stratified planning
│   ├── data.py               # RAM-cached dataset, D4(XY)xZ2(Z) symmetries, RNG checkpoint helpers
│   ├── models.py             # Zero-SNR schedule, ContinuousScalarLoadEmbedder, ContinuousLoadConditionedUNet, EMA
│   └── metrics.py            # Solid fraction, SSA, chords, anisotropy, FFT S2(r), percolation, geodesic tortuosity, OpenImpala export
├── scripts/
│   ├── preprocessing/
│   │   ├── segment_carbon_xct.py
│   │   └── extract_carbon_3d_patches.py
│   ├── training/
│   │   └── train_carbon_conditional_96.py
│   └── evaluation/
│       ├── sample_carbon_conditional_96.py
│       └── evaluate_carbon_physics.py
├── slurm/                    # Isambard-AI GH200 Slurm batch scripts
├── tests/                    # Automated pytest unit + end-to-end pipeline verification
├── experiments.md            # Chronological experimental log (Experiments 1–14)
├── pyproject.toml
├── environment.yml
└── requirements-isambard.txt
```

---

## Isambard-AI (GH200) Quickstart & Handover Guide

### 1. Environment Setup (one-time)

```bash
cd /projects/u6uc/asif/code/PorousDiff
git fetch origin
git checkout feature/paper-ready-conditional

source /projects/u6uc/asif/.venv/bin/activate
pip install --no-build-isolation --no-deps -e .

# Run local CPU verification suite (~45s) before submitting GPU jobs
pytest -v
```

> All scripts resolve data and results relative to `POROUSDIFF_ROOT` (defaults automatically to `/projects/u6uc/asif` when present, or current directory otherwise).

### 2. Calibrated Segmentation & Z-Artifact Gating

For the 3 currently staged datasets (`Op008mVV`, `Op018mVV`, `Op033mVV`), or any expanded list of Diamond synchrotron folders (`0p008mVV` through `0p053mVV` — both `data/raw/<name>/*.tif` and `data/raw/<name>/rec_16bit_Paganin/*.tif` are supported automatically):

```bash
sbatch slurm/segment_carbon_xct.slurm
# Or with custom staged folders:
sbatch --export=ALL,DATASETS="0p008mVV 0p013mVV 0p018mVV 0p023mVV 0p028mVV 0p033mVV 0p038mVV 0p043mVV 0p048mVV 0p053mVV" \
    slurm/segment_carbon_xct.slurm
```

Inspect `<POROUSDIFF_ROOT>/data/carbon_virgin/<dataset>_binary_corrected_carbon_calibrated_v1/z_quality_diagnostic.png`, `segmentation_report.json`, and `valid_slices.json` to confirm monotonic clean solid fractions and excluded ring/plunger intervals.

### 3. Z-Stratified `96³` Patch Extraction

```bash
sbatch slurm/extract_carbon_96.slurm
# Or with custom datasets:
sbatch --export=ALL,DATASETS="0p008mVV 0p013mVV 0p018mVV 0p023mVV 0p028mVV 0p033mVV 0p038mVV 0p043mVV 0p048mVV 0p053mVV" \
    slurm/extract_carbon_96.slurm
```

### 4. Training (Observed + Held-Out Interpolation Protocol)

Run a quick 4-step GPU smoke test first, then launch the 100-epoch `bf16` training job (if pre-empted or extended, resubmitting the **exact same** `train_carbon_conditional_96.slurm` automatically detects `latest.pth` and resumes):

```bash
# 3-state baseline verification:
sbatch slurm/test_carbon_conditional_96.slurm
sbatch slurm/train_carbon_conditional_96.slurm

# Paper interpolation experiment (train on 7 loads, hold out 0.018, 0.033, 0.048 mV/V):
sbatch --export=ALL,EXPERIMENT="carbon_cond96_interp_paper",STATES="0p008mVV=0.008 0p013mVV=0.013 0p023mVV=0.023 0p028mVV=0.028 0p038mVV=0.038 0p043mVV=0.043 0p053mVV=0.053" \
    slurm/train_carbon_conditional_96.slurm
```

### 5. Sampling, Continuous Compression Sweep & Physics Evaluation

Generates `N = 20` 3D realizations per load (both training and unseen held-out loads), runs a fixed-noise continuous compression sweep across `0.008..0.053 mV/V`, exports OpenImpala multi-page TIFF stacks, and produces `physics_validation_report.json`, `physics_compression_curves.png`, and `s2_correlation_comparison.png`:

```bash
# 3-state baseline evaluation:
sbatch slurm/sample_carbon_conditional_96.slurm

# Paper interpolation evaluation (highlighting unseen held-out loads):
sbatch --export=ALL,EXPERIMENT="carbon_cond96_interp_paper",EVAL_STATES="0p008mVV=0.008 0p013mVV=0.013 0p018mVV=0.018 0p023mVV=0.023 0p028mVV=0.028 0p033mVV=0.033 0p038mVV=0.038 0p043mVV=0.043 0p048mVV=0.048 0p053mVV=0.053",HELDOUT_STATES="0p018mVV 0p033mVV 0p048mVV" \
    slurm/sample_carbon_conditional_96.slurm
```

---

## Historical Benchmark Summary

Earlier unconditional baselines (`TiO₂`, `Op001mVV` / `Op026mVV`) established the motivation for the calibrated segmentation and Zero-SNR `v`-prediction formulation introduced in `v0.2` (full log in [`experiments.md`](experiments.md)):

| Experiment | Formulation | Patch Size | Real $\phi_s$ | Generated $\phi_s$ | Key Diagnostic Takeaway |
|---|---|---:|---:|---:|---|
| Exp 1–5 | 3D Conv Autoencoder (MSE + LPIPS) | `64³` | — | — | Reconstructs global envelope; smooths fine pores |
| Exp 6–11 | Grayscale 3D DDPM (`epsilon`, std linear) | `64³` | — | — | Blurry boundaries & horizontal banding |
| Exp 12 | Unconditional Binary DDPM (`epsilon`, std schedule) | `64³` | `0.1318` | `0.3948` | Non-zero terminal SNR shifts sparse binary mean ~3× toward `0.50` |
| Exp 13 | Unconditional Binary DDPM (`epsilon`, std schedule) | `96³` | `0.0984` | `0.2806` | `96³` feasible on GH200, but schedule/conditioning fixes required |
| **Exp 14 (`v0.2`)** | **Continuous Load-Conditioned DDPM (`v`-pred + Zero-SNR + EMA)** | **`96³`** | **Calibrated across `0.008–0.053 mV/V`** | **Physics-validated suite** | **Fixes Otsu inversion, ring/plunger seams, 1-token attention collapse & terminal-SNR bias** |

---

## Acknowledgements & License

Developed within [BASE Laboratory](https://github.com/BASE-Laboratory), with experiments performed on Isambard-AI NVIDIA GH200 computing resources (`brics.u6uc`). See [`LICENSE`](LICENSE) for licensing details.