Experiment 1:
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

Experiment 2:
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

Experiment 3:
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


Experiment 4:
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

    Experiment 5:
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

    