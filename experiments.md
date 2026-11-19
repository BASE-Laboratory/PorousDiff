## Experiment 1:
- 3D autoencoder baseline
- GH200 GPU
- 100 patches
- 5 epochs
- MSE loss
- Upsample + Conv3D decoder
- Loss decreased successfully
- Reconstruction captures global morphology but remains blurry
- Reconstruction captures large-scale morphology and intensity distribution
- Fine-grained microstructural details remain blurred
- Low MSE loss does not necessarily correspond to high perceptual fidelity
- Possible causes: shallow architecture, limited epochs, MSE smoothing effects
- Final training loss: 0.000804

## Experiment 2:
- Increased training duration from 5 epochs to 25 epochs
- Same 3D autoencoder baseline architecture
- GH200 GPU
- 100 patches
- MSE reconstruction loss
- Upsample + Conv3D decoder
- Training loss decreased significantly over epochs
- Final MSE loss reached very low values (~0.0008)
- Reconstruction remained blurry despite lower numerical loss
- Global morphology captured successfully
- Fine-grained microstructural details remain blurred
- Low MSE loss does not necessarily correspond to high perceptual fidelity
- Possible causes: shallow architecture, limited dataset size, MSE smoothing effects
- Demonstrated limitation of pixel-wise reconstruction objective

## Experiment 3:
- Increased raw dataset from 100 TIFF slices to 500 TIFF slices
- Generated 700 3D patches of size 64×64×64
- GH200 GPU
- 25 epochs
- MSE reconstruction loss
- Upsample + Conv3D decoder

Results:
- Training completed on larger patch dataset
- Loss curve and reconstruction figure saved
- Checkpoint saved

Observation:
- Compare reconstruction sharpness and loss behaviour against Experiment 2


## Experiment 4:
- Added LPIPS perceptual loss to baseline 3D autoencoder
- Combined loss function:
  Total Loss = MSE + 0.1 × LPIPS
- LPIPS applied slice-wise on middle 2D slices extracted from 3D volumes
- Grayscale slices converted to 3-channel format for LPIPS compatibility
- Dataset size: 700 patches generated from 500 TIFF slices
- Training epochs: 25
- GPU: NVIDIA GH200 120GB
- Decoder: Upsample + Conv3D architecture
- MSE loss continued decreasing steadily
- LPIPS loss decreased from ~0.21 to ~0.04
- Reconstruction quality improved noticeably compared to pure MSE training
- Fine structural textures and local morphology became more visible
- Reconstruction still slightly smooth but preserves significantly more microstructural detail
- Demonstrates perceptual-loss-based improvement over standard reconstruction objective
- Final losses:
    - Total Loss ≈ 0.00449
    - MSE Loss ≈ 0.00053
    - LPIPS Loss ≈ 0.03968

## Experiment 5:
- Increased dataset from 700 patches to 1200 patches
- Generated from 800 TIFF slices
- 3D autoencoder with deeper encoder-decoder architecture
- Combined MSE + LPIPS perceptual loss
- GH200 GPU training
- 25 epochs
- Reconstruction quality improved noticeably compared to earlier experiments
- Fine-grained textures and structural patterns are better preserved
- LPIPS loss decreased from ~0.20 to ~0.03 during training
- Reconstruction remains slightly blurry but captures morphology more accurately
- Larger dataset improved generalization and perceptual fidelity
- Final losses:
    - Total Loss ≈ 0.006
    - MSE ≈ 0.0007
    - LPIPS ≈ 0.056

## Experiment 5: MONAI 3D Diffusion Baseline

Status: setup test successful

- Loaded 1200 patches from data/processed/patches64
- Patch shape: 64×64×64
- CUDA available on GH200
- MONAI DiffusionModelUNet forward pass successful on real patch

Next:
- Add DataLoader
- Add DDPM noise scheduler
- Train model to predict noise

## Experiment 6: 3D DDPM baseline — 1 epoch test

Status: successful

- Dataset: 1200 real 64³ patches
- Model: MONAI DiffusionModelUNet
- GPU: NVIDIA GH200 120GB
- Batch size: 1
- Epochs: 1
- Average loss: 0.051106
- Checkpoint saved: results/diffusion_checkpoints/ddpm_3d_epoch1.pth

Next:
- Generate sample volumes from the checkpoint
- Save middle slices as PNG
- Then train for more epochs

## Experiment 7: 3D DDPM baseline — 25 epoch 

Status: successful

Configuration:
- Model: MONAI DiffusionModelUNet
- Dataset: 1200 real 64³ patches
- Batch size: 1
- Epochs: 25
- GPU: NVIDIA GH200 120GB

Results:
- Generated slices evolved beyond random noise
- Strong horizontal and anisotropic texture patterns emerged
- Model learned coarse spatial continuity across the volume
- Dark porous regions began appearing consistently

Observations:
- Generated outputs remained highly blurry
- Structures were dominated by low-frequency smooth textures
- Pore boundaries were not sharply defined
- Outputs resembled coarse learned texture fields rather than realistic porous microstructures

Interpretation:
The experiment demonstrated that the DDPM successfully learned early structural statistics of the porous dataset after 25 epochs, though realistic microstructure morphology had not yet emerged.

Outputs:
- Generated samples saved in:
  results/diffusion_samples/

- Checkpoint saved in:
  results/diffusion_checkpoints/

Next:
- Extend training to 50+ epochs
- Generate multiple samples for diversity analysis
- Improve normalization and model capacity
- Compare generated slices with real microstructure patches

## Experiment 8: 3D DDPM baseline — 50 epoch 

Status: successful

Configuration:
- Model: MONAI DiffusionModelUNet
- Dataset: 1200 real 64³ patches
- Batch size: 1
- Epochs: 50
- GPU: NVIDIA GH200 120GB

Results:
- Average loss decreased significantly during training
- Generated samples no longer resemble pure noise
- Spatial continuity and porous texture patterns emerged
- Samples show diversity without obvious mode collapse
- Early anisotropic/horizontal structures observed

Current limitations:
- Generated slices remain blurry
- Weak pore boundary definition
- Low-frequency smooth textures dominate
- Realistic porous connectivity not yet fully learned

Outputs:
- Checkpoint saved in:
  results/diffusion_checkpoints/

- Generated samples saved in:
  results/diffusion_samples/

Next steps:
- Train for 100 epochs
- Generate 20 samples
- Compare generated slices with real slices
- Evaluate porosity and structural realism
- Improve normalization and architecture if needed

## Experiment 9: 3D DDPM - 100 Epochs

Dataset:
- TiO2 patches
- ~1200 patches
- 64x64x64

Model:
- DiffusionModelUNet
- channels=(32,64,128)
- attention=(False,False,True)

Training:
- 100 epochs
- LR = 1e-4
- MSE loss

Results:
- Average final loss ≈ 0.0022
- Generated 20 samples

Observations:
- Samples exhibit realistic porous texture.
- Clear pore/solid contrast observed.
- Diversity improved compared to 25 and 50 epochs.
- Persistent horizontal streaking/banding visible in multiple generated samples.
- Possible causes:
  - dataset anisotropy
  - patch extraction orientation
  - limited training dataset size
  - model capacity limitations

Next Steps:
- Compare generated slices with original slices.
- Scale training to multiple materials.
- Investigate streaking artifacts reported in diffusion literature.


## Experiment 10: Improved 3D DDPM Architecture

Objective:

- Investigate whether modifying the 3D DDPM architecture reduces the horizontal streaking artifacts observed in the baseline TiO₂ DDPM outputs.

Initial Architecture Attempt:

- A larger DDPM architecture was first tested using increased channel width and additional attention:

```python
num_channels = (64, 128, 256)
attention_levels = (False, True, True)
num_res_blocks = 2
````

- This configuration failed due to CUDA out-of-memory on the NVIDIA GH200 GPU.
- The failure indicates that 3D diffusion models with high channel widths and attention layers are memory-intensive, even on high-memory accelerators.

Final Tested Architecture:

- To reduce memory cost, the original channel widths were retained.
- Network depth was increased by changing `num_res_blocks` from 1 to 2.
- Patch size remained 64×64×64.
- Dataset size remained approximately 1200 TiO₂ patches.
- Diffusion schedule, optimizer, learning rate, and training duration were kept the same as the baseline model.

Model Configuration:

```python
num_channels = (32, 64, 128)
attention_levels = (False, False, True)
num_res_blocks = 2
```

Training Results:

- Training completed successfully for 100 epochs.
- Final average loss: 0.001562.
- Checkpoints were saved every 10 epochs.
- Generated 20 synthetic samples for qualitative evaluation.

Observations:

- Horizontal streaking artifacts remained visible in the generated samples.
- Generated structures appeared smoother and blurrier than the baseline DDPM outputs.
- Increasing residual depth alone did not improve the reproduction of elongated TiO₂ particle morphology.
- The generated samples appeared to capture low-frequency intensity bands rather than fine microstructural geometry.
- The result suggests that model depth alone is not the primary factor causing the artifact.

Conclusion:

- Increasing `num_res_blocks` from 1 to 2 was insufficient to reduce horizontal streaking.
- A larger channel-width model with additional attention was not feasible due to GPU memory limitations.
- These results suggest that the main bottleneck may be related to patch size, dataset diversity, preprocessing strategy, or the training data itself rather than only network depth.
- Future experiments will focus on training separate and combined TiO₂ datasets, followed by multi-material training to investigate whether increased data diversity reduces the artifact.


## Experiment 11: Baseline DDPM on Op026mVV

Objective:

- Evaluate whether a different TiO₂ tomography dataset exhibits the same generation artifacts observed in Op001mVV.

Dataset:

- Op026mVV TiO₂
- 800 TIFF slices
- ~1200 patches
- Patch size: 64×64×64

Model:

- Baseline DDPM
- num_channels=(32,64,128)
- num_res_blocks=1
- 100 epochs

Training Results:

- Final average loss ≈ 0.0028
- Stable convergence throughout training
- Checkpoints saved every 10 epochs

Observations:

- Horizontal streaking artifacts were reduced compared with Op001.
- Generated samples appeared more isotropic and blob-like.
- Fine microstructural details remained absent.
- Samples remained blurry and lacked clear particle boundaries.

Conclusion:

- Dataset characteristics appear to influence generated morphology.
- Artifact reduction was achieved without architectural changes.
- Results suggest data diversity may have greater impact than increasing network depth.

## Experiment 12: Binary 3D DDPM on Op001mVV — 64³

**Status:** Completed — preliminary physical evaluation

### Objective
Investigate whether binary segmentation improves 3D diffusion generation compared with the earlier grayscale experiments.

### Dataset and preprocessing
- Material: TiO₂ (Op001mVV)
- Input: 800 XCT TIFF slices
- Background correction: Gaussian filtering (sigma = 30)
- Intensity normalization: percentile clipping (1st–99th)
- Segmentation: Otsu thresholding
- Binary convention: 1 = solid, 0 = pore
- Patch size: 64 × 64 × 64
- Training patches: 960
- Validation patches: 240

### Model and training
- Architecture: MONAI DiffusionModelUNet
- Channels: (32, 64, 128)
- Attention levels: (False, False, True)
- Residual blocks: 1
- Diffusion timesteps: 1000
- Optimizer: AdamW
- Learning rate: 0.0001
- Batch size: 1
- Training epochs: 100

### Results
- Final training loss: 0.004893
- Final validation loss: 0.003762
- Generated samples evaluated: 10
- Mean real training-patch solid fraction: 0.1318
- Mean generated solid fraction: 0.3948

### Observations
- The model completed 100 epochs and generated binary 3D volumes.
- Generated samples substantially overestimated the solid fraction.
- Preliminary morphological comparisons indicated differences in spatial structure and connectivity.
- Low noise-prediction loss did not guarantee physically representative microstructures.

### Conclusion
Binary segmentation enabled a more direct comparison of pore and solid phases, but the generated volumes did not reproduce the training solid-fraction distribution.

Physical fidelity remains insufficient for claiming realistic digital twins.


## Experiment 13: Binary 3D DDPM on Op001mVV — 96³

**Status:** Completed — preliminary physical evaluation

### Objective
Investigate whether increasing patch size from 64³ to 96³ improves the fidelity of binary 3D microstructure generation.

### Dataset
- Material: TiO₂ (Op001mVV)
- Input: 800 segmented XCT TIFF slices
- Patch size: 96 × 96 × 96
- Training patches: 960
- Validation patches: 240
- Binary convention: 1 = solid, 0 = pore

### Model and training
- Architecture: MONAI DiffusionModelUNet
- Channels: (32, 64, 128)
- Attention levels: (False, False, True)
- Residual blocks: 1
- Diffusion timesteps: 1000
- Optimizer: AdamW
- Learning rate: 0.0001
- Batch size: 1
- Training epochs: 100
- GPU: NVIDIA GH200

### Results
- Final training loss: 0.003370
- Final validation loss: 0.003983
- Training duration: approximately 9 hours 46 minutes
- Generated samples evaluated: 10
- Mean real training-patch solid fraction: 0.0984
- Mean generated solid fraction: 0.2806
- Generated solid-fraction range: 0.0438–0.8830

### Observations
- Training and inference completed successfully.
- Generated solid fractions varied substantially across samples.
- Some generated samples had solid fractions close to the training mean.
- Real-versus-generated orthogonal views nevertheless showed notable morphological differences, including fragmented solid structures.
- Larger patch size alone did not resolve physical-fidelity limitations.

### Limitations
- Only 10 generated volumes were evaluated.
- The 64³ and 96³ training datasets had different solid-fraction distributions, limiting direct comparison.
- Full 3D connectivity, spatial-correlation and tortuosity validation remain incomplete.
- Segmentation quality requires further validation against the original XCT data.

### Conclusion
Increasing patch size to 96³ was computationally feasible on the GH200 but did not consistently reproduce the physical characteristics of the training microstructures.

Future work will investigate compression-conditioned generation using CuCHF datasets with three experimentally measured load-cell conditions.


## Experiment 14: Synchrotron XCT Diagnostic Audit & Continuous Load-Conditioned 3D DDPM (`v0.2`)

**Status:** Implemented and verified end-to-end (`feature/paper-ready-conditional`) — ready for full Isambard-AI GH200 paper runs

### Diagnostic Findings on Raw Carbon-Felt XCT Volumes (`2160 × 2560 × 2560` `uint16` Paganin)
Direct inspection of raw synchrotron compression scans (`crbon_virgin_1_0p008mVV` through `0p053mVV`) identified five root causes behind earlier binary/conditional instabilities:
1. **Compression axis orientation:** `Z` (slice index `0..2159`) is the vertical mechanical compression plunger axis; `XY` is the carbon-felt sheet plane.
2. **Circular punch geometry:** The cylindrical carbon-felt punch (`R ≈ 950 px` centered at `(1280, 1280)`) sits inside square `2560 × 2560` reconstructions surrounded by gray exterior air (`~30,000` counts). Unmasked square grids extracted exterior air corners.
3. **Plunger intrusion & mid-stack bullseye ring seams:** Under higher loads, the bottom compression plunger rises into the stack (`Z ≈ 1850+` at `0.028 mV/V`; `Z ≈ 1280+` at `0.053 mV/V`), and non-physical concentric ring / stitch interference bands appear mid-stack (including `Z ≈ 520–620` inside the first `800` slices at `0.053 mV/V`).
4. **Per-slice Otsu inversion on sparse low-load slices:** Per-slice `[1, 99]` percentile clipping + `sigma = 50` Gaussian subtraction + per-slice Otsu inverted empty pore noise into ~80% false solid on sparse `0.008 mV/V` slices and erased true monotonic compression trends.
5. **1-token cross-attention degeneracy & non-zero terminal SNR:** Passing a 1-token context `[B, 1, 32]` into cross-attention forced `softmax([s]) = 1.0` identically (zero gradient to `W_Q, W_K`), while standard linear/scaled-linear schedules left residual signal at `t = T - 1` that biased sparse `{−1, +1}` microstructures upward toward `50%` solid fraction.

### Architectural & Pipeline Fixes (`porousdiff/`)
- **Calibrated segmentation (`segment_carbon_xct.py`):** Circular ROI mask (`radius_fraction = 0.70`), pore-mode-clamped background subtraction (`sigma = 32`), shared `delta_threshold = 5500` across all loads (recovering clean monotonic solid fractions `~5.4%` at `0.008 mV/V` → `~23.5%` at `0.028 mV/V` → `~33.8%` at `0.053 mV/V`), and automatic Z-interval gating (`compute_radial_ring_score` + slice-jump + pore-mode drift).
- **Z-stratified patch extraction (`extract_carbon_3d_patches.py`):** Round-robin sampling across all valid Z-windows, `>= patch_size` buffer between `train` and `val` Z-slabs, and stacked `patches.npy` archives.
- **Multi-scale continuous conditioning (`ContinuousLoadConditionedUNet`):** Smooth Fourier scalar load embedder (`ContinuousScalarLoadEmbedder`) injected into `temb` at every ResBlock across all resolution levels plus 3-channel spatial input `[x_t, cond_norm * mask, mask]`, Zero-Terminal-SNR schedule (`zero_snr_scaled_linear`), `v_prediction`, EMA (`0.9995`), compression-safe `D_4(XY) × Z_2(Z)` augmentation, and `bf16` GH200 mixed precision.
- **Physics & transport evaluation (`evaluate_carbon_physics.py`):** Automated comparison of solid fraction, specific surface area, directional chord lengths, `XY/Z` compression anisotropy, directional $S_2(r)$, 3D percolation, geodesic pore tortuosity, and OpenImpala multi-page TIFF export across both training and held-out interpolated loads.


## Experiment 15: Multi-Electrode, 3-Phase Few-Shot Transfer Architecture (`v0.3`)

**Status:** Software pipeline implemented and verified on synthetic test suites (`feature/paper-ready-conditional`) — awaiting full GH200 evaluation on Virgin Carbon, $\text{TiO}_2$ Anode, CuHCF Cathode, and DLS K11 DIAD datasets

### Scientific Motivation & Literature Gap
Existing 3D diffusion papers for porous electrodes generate static microstructures or condition on global porosity, while recent compression-transport studies (*ACS Appl. Energy Mater.* 2025) rely on heuristic numerical Z-squashing or column-wise pore-voxel deletion. Across our multi-electrode synchrotron compression suite—**Virgin Carbon Felt** (`virgin_carbon`), **$\text{TiO}_2$-coated Carbon Felt Anode** (`tio2_anode`), and **Copper Hexacyanoferrate Coated Cathode** (`cuhcf_cathode`)—plus an independent cross-resolution repeat at **Diamond Light Source K11 DIAD**, `PorousDiff` implements a held-out cross-electrode benchmark.

### Key `v0.3` Upgrades (`porousdiff/`)
1. **3-Phase Solid-Masked Calibration & Coarse-Grained Binning (`preprocessing.py`):**
   - Two-stage Otsu calibration (`calibrate_multiphase_thresholds` + `segment_slice_multiphase`) separates `0 = pore`, `1 = carbon_fiber`, and `2 = active_coating` ($\text{TiO}_2$ / CuHCF) without splitting the dominant 80–90% pore background peak.
   - Volume-fraction-matched `downsample_volume_rev` (`spatial_bin = 2` or `3`, paired with `audit_binning_topology_preservation`) enlarges $96^3$ patches to $2\times$–$3\times$ physical field-of-view (`~20–30` fiber diameters per patch edge).
   - `CalibrationMetadata` records physical `voxel_size_um`, `kpa_per_mvv` mechanical stress conversion, `beamline`, and `is_repeat` in `bundle_manifest.json`, plus a ~10-second `--preview-only` Spyder mode (`qa/multiphase_overlay_diagnostic.png`).
2. **Literature Numerical Z-Compression Baselines (`baselines.py`):**
   - `affine_z_compress_to_target` (affine Z-resampling + occupancy-quantile projection matching target compressed porosity) and `pillar_pore_deletion_compress` (stochastic XY-column pore-layer removal).
3. **Hierarchical 2-Channel 3-Phase Representation & Multi-Modal Conditioning (`data.py`, `models.py`):**
   - Hierarchical encoding `[c_solid, c_coating]` in `{-1, +1}` aligns Channel 0 (`total solid vs. pore`) across 2-phase Virgin Felt and 3-phase coated electrodes while `decode_channels_to_phases` prevents floating unattached coating artifacts.
   - `MultiModalConditionEmbedder` embeds normalized stress/load + categorical `material_id` (`nn.Embedding`) + continuous `log(dx / dx_ref)` voxel size into `temb` and spatial conditioning channels.
4. **3-Phase Electrochemical Interfacial Metrics & Held-Out Benchmark (`metrics.py`, `evaluate_beamtime_multiplier.py`):**
   - Computes Electrochemically Active Surface Area (`ssa_pore_coating`), Triple-Phase Boundary line density (`tpb_density` in $\mu\text{m}^{-2}$), electronically wired coating fraction (`wired_coating_fraction`), 26-connected Euclidean geodesic tortuosity, and multi-phase OpenImpala TIFF export (`{0, 1, 2}`).


## Experiment 16: Conditional DDIM Transfer, MultiDiffusion Tiling, $\hat{x}_0$ Regularization & OpenImpala Clamping-Pressure ($\sigma^*$) Pipeline (`v0.4`)

**Status:** Software pipeline & peer-review audit fixes verified locally and in CI (`feature/paper-ready-conditional`)

### Key `v0.4` Capabilities & Methodological Safeguards (`porousdiff/`)
1. **Conditional DDIM Microstructure Transfer & Fidelity Audit (`porousdiff/virtual_compression.py`, `scripts/evaluation/virtual_compress_scan.py`):**
   - Partially inverts a real low-load segmented 3D scan (`invert_strength = 0.60`, $\bar{\alpha} \approx 0.16$) along the conditional DDIM trajectory and denoises to higher in-range target stresses $\sigma$. Automatically evaluates `check_load_extrapolation`, roundtrip `same_load_solid_dice`, and `paired_xy_column_corr` vs. unpaired scans (`xy_correlation_lift`) to distinguish specimen-specific structural retention from unconditional resampling.
2. **Seamless $>96^3$ 3D Volume Stitching (`generate_tiled_3d`, `--tiled-shape`):**
   - Symmetrically spaced 3D raised-cosine (Hann) MultiDiffusion synthesizes larger domains (e.g. $96 \times 96 \times 192$ or $160^3$) so REV domain-size convergence can be tested explicitly in OpenImpala.
3. **Auxiliary $\hat{x}_0$ Training Penalties & Coating-Retention Audit (`porousdiff/losses.py`, `--physics-loss`):**
   - Closed-form single-step $\hat{x}_0$ reconstruction paired with Min-SNR-$\gamma$ weighting, 3D finite-difference interfacial sharpness, hierarchical no-floating-coating support penalty $\text{ReLU}(\hat{c}_{\text{coating}} - \hat{c}_{\text{solid}})$, and solid-phase catalyst-to-scaffold ratio conservation, plus `val_x0_sf_mae` validation logging and `coating_retention_diagnostics.json`.
4. **Clamping-Pressure $\sigma^*$ Analysis via 1D Newman ASR & Weight Sensitivity (`porousdiff/optimization.py`, `scripts/evaluation/optimize_clamping_pressure.py`):**
   - Computes both a normalized harmonic Figure of Merit (accompanied by an 11-point simplex `weight_sensitivity` sweep) and a dimensional **1D Newman & Tobias porous-electrode Area-Specific Resistance $R_{\text{ASR}}(\sigma)$ ($\Omega\cdot\text{cm}^2$)** incorporating compressed thickness $L(\sigma) = L_0(1 - \epsilon_Z)$, Nernst diffusive mass-transfer resistance, clamping contact resistance, Darcy hydraulic penalty, and 3-regime sensitivity (`kinetics_limited`, `balanced`, `mass_transport_limited`). Refuses to mix partial OpenImpala + surrogate estimates on the same curve.
5. **Partial-Correlation DIAD XRD & Pinch-Point Analysis (`porousdiff/metrics.py`, `porousdiff/transport_and_xrd.py`):**
   - Quantifies `specific_z_pinch_excess` (`P(solid[z] & solid[z-1]) / phi_s^2`), `coating_pinch_fraction`, and `mean_coating_coordination`, reporting both raw Pearson $r$ and OLS partial correlation `partial_pearson_r_given_sf` (controlling for solid fraction $\phi_s$) against co-registered DLS K11 DIAD XRD microstrain.
6. **Peer-Review & Red-Team Engineering Hardening (`v0.4`):**
   - **Single Load-Unit Source of Truth:** `sample_carbon_conditional_96.py`, `evaluate_beamtime_multiplier.py`, and `optimize_clamping_pressure.py` read `use_stress_kpa` directly from checkpoint config (`resolve_use_stress_kpa_from_checkpoints`) and resolve loads via `resolve_scan_load_value` (preserving `0.0 kPa` and raising on missing units rather than falling back to `mV/V`).
   - **Resume Data-Leakage Guard (`validate_resume_config`):** `train_carbon_conditional_96.py --resume` verifies `heldout_states`, `include_states`, `use_stress_kpa`, `states`, and `scan_specs` match the saved checkpoint before loading weights.
   - **Mirror-Free Tall Baseline Sub-Volumes (`val/tall_patches.npy`):** `prepare_local_drive_bundle.py` extracts full-clean-height validation columns so `affine_z_compress_to_target` and `pillar_pore_deletion_compress` operate with `baseline_mirror_tiled_fraction == 0.0` whenever clean scan height permits.
   - **26-Connected Euclidean Geodesic Tortuosity:** Upgraded `geodesic_pore_tortuosity` from 6-neighbour Manhattan BFS to 26-neighbour Euclidean Dijkstra (`1, sqrt(2), sqrt(3)` step weights, recovering $\tau = \sqrt{2}$ on $45^\circ$ diagonal channels).
   - **3-Phase Partial-Volume Halo Cleanup & Thin-Scan XY Split:** Strips 1–2 px Paganin/PSF phase-`1` blur rinds at `0–2` pore boundaries (`strip_partial_volume_carbon_halo`) and falls back to buffered XY-sector splitting (`split_xy_positions_with_buffer`) on thin compressed stacks.


## Experiment 17: Native `OpenImpala` v4.6 Python Coupling, Unbiased Cauchy–Crofton Morphology, Constriction $\beta_Z(\sigma)$ & Mass-Conserving Lagrangian Slab DDIM (`v0.5`)

**Status:** Software tracks `v0.5a`–`v0.5d` implemented and verified with `OpenImpala` v4.6 in CI (`feature/paper-ready-conditional`) — staged for Isambard-AI GH200 production campaign (`v1.0`)

### Key `v0.5` Upgrades (`porousdiff/`)
1. **In-Process `OpenImpala` v4.6 Python Facade Integration (`porousdiff/transport_and_xrd.py`, `evaluate_beamtime_multiplier.py --run-openimpala`):**
   - `compute_openimpala_volume_transport` and `run_openimpala_python_suite` execute steady-state PDE transport directly on `(Z, Y, X)` NumPy arrays (`openimpala.tortuosity_all(..., strict=False)` and optional $3 \times 3$ `openimpala.effective_diffusivity`) across ionic pores (`phase=0`), graphitic carbon (`phase=1`), active coating (`phase=2`), and composite 3-phase solid conduction (`phase_diffusivities={0: 0.0, 1: 1.0, 2: sigma_coat_rel}`), recording `d_eff = 0.0` cleanly on pinched non-percolating directions.
   - `run_openimpala_rev_convergence` (`sample_carbon_conditional_96.py --run-rev-study`) wraps `openimpala.rev_study` + `through_thickness_profile` to audit sub-volume variance decay ($\text{CoV}(D_{zz}) \le 5\%$) and Z-direction compaction uniformity.
2. **Cauchy–Crofton $2/3$ Stereology, Continuum Constriction $\beta_Z(\sigma)$, 3D Euler Genus & Ensemble $W_1$ (`metrics.py`, `optimization.py`):**
   - Emits isotropic $2/3$ Cauchy–Crofton interfacial areas (`ssa_total_cc`, `ssa_pore_coating_cc`, `ssa_pore_carbon_cc`, `ssa_carbon_coating_cc`) matching `openimpala.specific_surface_area(..., ssa_corrected)`, and uses Cauchy–Crofton kinetic area by default in 1D Newman & Tobias $R_{\text{ASR}}(\sigma)$ (`"kinetic_area_convention": "cauchy_crofton_2_3"`).
   - Computes continuum pore-throat constriction $\beta_Z(\sigma) = \left(D_{\text{eff},Z}^{\text{PDE}} / D_0\right) \tau_{\text{geo},Z}^2(\sigma) / \varepsilon(\sigma)$, exact digital-topology 3D cubical Euler characteristic density $\chi_V = (V - E + F - C) / (N \Delta x^3)$, EDT inscribed pore radii (`pore_radius_median_um`, `pore_radius_p90_um`), and ensemble Wasserstein-1 ($W_1$) / two-sample KS fidelity distances.
3. **Mass-Conserving Lagrangian Slab Virtual Compression (`virtual_compress_lagrangian_slab`, `virtual_compress_scan.py --mode lagrangian_slab`):**
   - Maps tall uncompressed scans `(Z_0, Y, X)` via volume-fraction-matched affine Z-compression to physical compressed height $Z(\sigma)$ and refines local morphology via overlapped 3D Hann MultiDiffusion DDIM (`refine_strength = 0.35`) with solid/coating mass conservation.
4. **Stage-Gated Isambard-AI GH200 Paper Campaign (`v1.0` Milestone):**
   - **Exp 17a (Dense Single-Material Baselines):** 3 seeds (`42, 43, 44`) $\times$ Virgin Carbon, $\text{TiO}_2$ Anode, CuHCF Cathode (`--physics-loss` ablation on/off).
   - **Exp 17b (Beamtime Multiplier Few-Shot Transfer):** Dense Virgin Carbon + `{2, 3}` sparse coated loads predicting held-out intermediate loads vs. `noop_uncompressed`, `affine_z_squash`, `pillar_pore_deletion`, Eulerian DDIM, and Lagrangian Slab DDIM.
   - **Exp 17c (Full OpenImpala GH200 PDE + K11 DIAD XRD Closure):** Compiled C++/CUDA OpenImpala $3 \times 3$ $D_{\text{eff}}$ & REV study ($96^3 \to 192^3$), constriction factor $\beta_Z(\sigma)$, Newman $R_{\text{ASR}}(\sigma)$ $\sigma^*$ with bootstrap CIs, and partial correlation against K11 DIAD microstrain.


