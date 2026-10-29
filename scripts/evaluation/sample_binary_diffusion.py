"""
sample_diffusion.py

Generate synthetic 3D battery electrode microstructure samples
from a trained 3D DDPM checkpoint.

Project: PorousDiff
code: Asif Ali
"""

from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler
from generative.inferers import DiffusionInferer


# -----------------------------
# Configuration
# -----------------------------

epoch = 5
num_samples = 1
sample_shape = (1, 1, 64, 64, 64)

checkpoint_path = Path("/projects/u6uc/asif/checkpoints/binary_pilot/binary_ddpm_3d_epoch5.pth")
output_dir = Path("/projects/u6uc/asif/results/binary_pilot/samples")

output_dir.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Device setup
# -----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# -----------------------------
# Load trained DDPM model
# -----------------------------

model = DiffusionModelUNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    num_channels=(32, 64, 128),
    attention_levels=(False, False, True),
    num_res_blocks=1,
).to(device)

model.load_state_dict(
    torch.load(checkpoint_path, map_location=device, weights_only=True)
)

model.eval()

print(f"Loaded checkpoint: {checkpoint_path}")


# -----------------------------
# DDPM scheduler and inferer
# -----------------------------

scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    schedule="scaled_linear_beta",
)

inferer = DiffusionInferer(scheduler)


# -----------------------------
# Generate synthetic samples
# -----------------------------

for i in range(num_samples):
    # Start generation from random Gaussian noise
    noise = torch.randn(sample_shape).to(device)

    # Reverse diffusion process:
    # noise -> synthetic 3D microstructure volume
    with torch.no_grad():
        sample = inferer.sample(
            input_noise=noise,
            diffusion_model=model,
            scheduler=scheduler,
        )

    # Remove batch and channel dimensions
    # Shape: [1,1,64,64,64] -> [64,64,64]
    sample = sample[0, 0].detach().cpu().numpy()


    # Model output is in approximately [-1, 1].
    # Convert to [0, 1], then threshold for binary evaluation.
    probability = np.clip((sample + 1.0) / 2.0, 0.0, 1.0)
    binary = (probability >= 0.5).astype(np.uint8)

    np.save(output_dir / f"binary_sample_{i+1:03d}.npy", binary)
    np.save(output_dir / f"continuous_sample_{i+1:03d}.npy", probability)

    mid = binary.shape[0] // 2

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    views = [
        (binary[mid, :, :], "XY"),
        (binary[:, mid, :], "XZ"),
        (binary[:, :, mid], "YZ"),
    ]

    for ax, (view, title) in zip(axes, views):
        ax.imshow(view, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle("Binary DDPM — 5 Epoch Pilot")
    fig.tight_layout()

    output_path = output_dir / f"binary_sample_{i+1:03d}_orthogonal.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    print(f"Saved binary sample {i+1}: {output_path}", flush=True)
    print(f"Generated solid fraction: {binary.mean():.4f}", flush=True)
