# PorousDiff

**Generative modelling of three-dimensional porous electrode microstructures using diffusion models.**

Developed by [BASE Laboratory](https://github.com/BASE-Laboratory) using NVIDIA GH200 accelerators on Isambard-AI.

**Research status:** Active development — TiO₂ unconditional 3D diffusion benchmarks completed; carbon-felt load-cell-conditioned 3D diffusion implemented and trained for 28 epochs. Physical validation remains ongoing.

## Overview

PorousDiff investigates the use of generative artificial intelligence to produce synthetic three-dimensional electrode microstructures from X-ray computed tomography (XCT) data.

The long-term objective is to develop a physics-validated generative framework capable of modelling microstructural changes under mechanical compression.

Current experiments focus on TiO₂ electrode microstructures and evaluate whether 3D denoising diffusion probabilistic models (DDPMs) can reproduce realistic pore and solid structures.

## Research Objectives

1. Develop reproducible preprocessing pipelines for XCT datasets.
2. Train and benchmark 3D autoencoder and diffusion models.
3. Evaluate generated microstructures using physically meaningful metrics.
4. Investigate how preprocessing, patch size, and architecture affect generative fidelity.
5. Extend the framework to compression-conditioned generation using CuCHF datasets.

## Implemented Features

- XCT TIFF inspection and preprocessing.
- Background correction and binary segmentation.
- 3D patch extraction with training and validation splits.
- MONAI-based 3D DDPM training.
- Checkpointing and training resumption.
- Slurm-based GPU training on Isambard-AI.
- Generation of synthetic 3D volumes.
- Orthogonal XY, XZ, and YZ visualization.
- Preliminary solid-fraction and morphological evaluation.

## Experimental Progress

### 3D Autoencoder Baselines

Initial experiments investigated convolutional 3D autoencoders using mean squared error (MSE) and slice-wise learned perceptual image patch similarity (LPIPS) losses.

Perceptual-loss experiments improved qualitative reconstruction detail compared with MSE-only baselines, although reconstruction smoothing remained.

### Grayscale 3D Diffusion Models

MONAI-based DDPMs were trained on 64³ XCT patches from TiO₂ datasets, including Op001mVV and Op026mVV.

Training was completed for up to 100 epochs.

Generated volumes developed porous textures but exhibited limitations, including horizontal banding, blurred boundaries, and incomplete reproduction of fine-scale morphology.

Increasing the number of residual blocks did not eliminate these artifacts.

### Binary 3D Diffusion Models

Binary segmentation was introduced to support more direct evaluation of pore and solid structures.

Two Op001mVV experiments were completed:

| Configuration | 64³ DDPM | 96³ DDPM |
|---|---:|---:|
| Training patches | 960 | 960 |
| Validation patches | 240 | 240 |
| Training epochs | 100 | 100 |
| Final training loss | 0.004893 | 0.003370 |
| Mean real solid fraction | 0.1318 | 0.0984 |
| Mean generated solid fraction | 0.3948 | 0.2806 |
| Generated samples evaluated | 10 | 10 |

**Main finding:** Successful DDPM training and decreasing noise-prediction loss did not guarantee physically representative synthetic microstructures.

The generated volumes showed substantial differences in solid fraction and morphology compared with real XCT patches.

The 64³ and 96³ datasets also had different solid-fraction distributions, so the results do not establish a controlled patch-size comparison.

These findings are preliminary. Comprehensive connectivity, spatial-correlation, and transport-property validation remains incomplete.

Detailed experiment records are available in [experiments.md](experiments.md).

### Carbon-Felt Conditional 3D Diffusion

A load-cell-conditioned 3D DDPM has been implemented for carbon-felt XCT microstructures under three experimental loading conditions.

**Dataset and preprocessing:**

- Conditions: 8, 18, and 33 mV load-cell readings.
- These readings are sensor measurements, not calibrated compression ratios.
- Binary XCT patches: 96 x 96 x 96 voxels (1 = solid, 0 = pore).
- Training patches: 2,880 (960 per condition).
- Validation patches: 720 (240 per condition).
- Preprocessing includes XCT segmentation and 3D patch extraction.

**Model and training:**

- MONAI Generative Models `DiffusionModelUNet`.
- 3D UNet channels: (32, 64, 128).
- Conditional cross-attention with a 32-dimensional embedding of the load-cell reading.
- DDPM noise schedule: 1,000 timesteps.
- Batch size: 1; AdamW optimizer; learning rate: 1e-4.
- NVIDIA GH200 GPU training using Slurm.
- Checkpoint saving and resumption implemented and tested.

**Recorded progress:**

| Metric | Value |
|---|---:|
| Completed epochs | 28 |
| Epoch 28 training loss | 0.006122 |
| Epoch 28 validation loss | 0.005096 |
| Training patches | 2,880 |
| Validation patches | 720 |

The losses measure noise-prediction performance, not physical fidelity.
The conditional model remains under evaluation; additional generation,
morphological analysis, and physics-based validation are required.

**Relevant scripts:**

- `scripts/preprocessing/segment_carbon_xct.py`
- `scripts/preprocessing/extract_carbon_3d_patches.py`
- `scripts/training/train_carbon_conditional_96.py`
- `scripts/evaluation/sample_carbon_conditional_96.py`
- `slurm/train_carbon_conditional_96.slurm`
- `slurm/resume_carbon_conditional_100.slurm`

## Technical Stack

- **Language:** Python
- **Deep learning:** PyTorch
- **Generative modelling:** MONAI Generative Models
- **Scientific computing:** NumPy and SciPy
- **Image processing:** tifffile and scikit-image
- **Visualization:** Matplotlib
- **HPC:** Slurm and NVIDIA GH200 GPUs

## Repository Structure

```text
PorousDiff/
├── configs/
├── docs/
├── scripts/
│   ├── preprocessing/
│   ├── training/
│   └── evaluation/
├── slurm/
├── experiments.md
├── environment.yml
├── requirements-isambard.txt
└── README.md
```

## Getting Started

Clone the repository:

```bash
git clone https://github.com/BASE-Laboratory/PorousDiff.git
cd PorousDiff
```

The repository includes environment specifications and experimental scripts. Some Slurm scripts contain cluster-specific paths and settings that must be adapted before use on another HPC system.

Research datasets and trained checkpoints are not distributed through this repository.

## Next Steps: Conditional Generation and Validation

The carbon-felt conditional 3D DDPM has been implemented and trained for 28 epochs using three load-cell readings (8, 18, and 33 mV).

Remaining research objectives include:

- Continue training when GPU resources become available.
- Evaluate conditional samples at all three loading conditions.
- Compare generated and real solid-fraction distributions.
- Assess connectivity, directional anisotropy, and spatial correlations.
- Investigate physical transport properties and model generalization.
- Improve portable inference and reproducibility documentation.

The current results are preliminary and do not yet establish physically validated digital twins.

## Limitations

Current generated microstructures have not consistently reproduced the solid-fraction distributions or spatial morphology of the XCT training data.

Additional validation is required before using generated volumes as physically representative electrode digital twins.

## Acknowledgements

Developed within BASE Laboratory, with experiments performed using Isambard-AI computing resources.

## License

See [LICENSE](LICENSE) for licensing information.