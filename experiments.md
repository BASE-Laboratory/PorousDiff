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