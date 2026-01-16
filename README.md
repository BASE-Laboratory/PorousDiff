# PorousDiff

> **Generative Surrogate Modelling for Mechanical Deformation in Aqueous Al-Ion Battery Electrodes.**

[![Status](https://img.shields.io/badge/Status-Pre--Alpha-red)]()
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![Compute](https://img.shields.io/badge/Compute-Isambard--AI%20(GH200)-green)]()

**PorousDiff** is a conditional 3D diffusion model designed to act as a Virtual Compression Rig for anisotropic porous media. 

Developed by the **[BASE Laboratory](https://github.com/BASE-Laboratory)** for the **Isambard-AI** supercomputer.

---

### 🔬 The Problem
Optimising electrode density requires understanding how fibrous microstructures buckle and close pores under compression.
* **Synchrotron Experiments:** Accurate but prohibitively expensive and slow.
* **DEM/FEM Simulations:** Computationally intractable for full-scale Representative Elementary Volumes (REVs).
* **Standard GANs:** Fail to capture long-range anisotropic connectivity (through-plane tortuosity).

**PorousDiff** solves this by learning a continuous mapping between a scalar compression ratio $C$ and the resulting 3D microstructure.

### 🛠️ Technical Stack
This project will use a high-performance stack designed for the NVIDIA Grace-Hopper Superchip (GH200).

* **Core Architecture:** 3D Denoising Diffusion Probabilistic Model (DDPM)
* **Conditioning:** Feature-wise Linear Modulation (FiLM) for scalar physics injection.
* **Framework:** [MONAI Generative Models](https://github.com/Project-MONAI/GenerativeModels) (PyTorch).
* **Configuration:** [Hydra](https://hydra.cc/) (structured `.yaml` configs).
* **Experiment Tracking:** [Weights & Biases](https://wandb.ai/).
* **Validation:** [OpenImpala](https://github.com/BASE-Laboratory/OpenImpala) (Tortuosity/Transport).

### 🚀 Getting Started (For Developers)

**1. Clone the Repository**
```bash
git clone [https://github.com/BASE-Laboratory/PorousDiff.git](https://github.com/BASE-Laboratory/PorousDiff.git)
cd PorousDiff
