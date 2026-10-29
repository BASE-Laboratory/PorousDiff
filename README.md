# PorousDiff

**Generative modelling of three-dimensional porous electrode microstructures using diffusion models.**

Developed by [BASE Laboratory](https://github.com/BASE-Laboratory) using NVIDIA GH200 accelerators on Isambard-AI.

**Research status:** Active development — unconditional 3D diffusion benchmarks completed; compression-conditioned generation planned.

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

## Planned Work: Compression-Conditioned Generation

The next research phase will investigate CuCHF electrode microstructures acquired under three experimental load-cell conditions.

The proposed conditional diffusion model will use an experimental condition as an input to generate synthetic microstructures corresponding to that condition.

Planned developments include:

- Conditioning on measured load-cell signals.
- Generation across three observed experimental states.
- Investigation of larger 3D patches.
- Physical validation of generated structures.
- Portable inference from trained model checkpoints.

These capabilities are research objectives and are not yet presented as completed implementations.

## Limitations

Current generated microstructures have not consistently reproduced the solid-fraction distributions or spatial morphology of the XCT training data.

Additional validation is required before using generated volumes as physically representative electrode digital twins.

## Acknowledgements

Developed within BASE Laboratory, with experiments performed using Isambard-AI computing resources.

## License

See [LICENSE](LICENSE) for licensing information.