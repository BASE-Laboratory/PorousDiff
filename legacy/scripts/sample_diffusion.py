"""
sample_diffusion.py

Generate synthetic 3D battery electrode microstructure samples
from a trained 3D DDPM checkpoint.

Project: PorousDiff
code: Asif Ali
"""

from pathlib import Path

import torch
import matplotlib.pyplot as plt

from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler
from generative.inferers import DiffusionInferer


# -----------------------------
# Configuration
# -----------------------------

epoch = 100
num_samples = 20
sample_shape = (1, 1, 64, 64, 64)

checkpoint_path = Path(f"results/diffusion_checkpoints/ddpm_3d_epoch{epoch}.pth")
output_dir = Path("results/diffusion_samples")

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
    torch.load(checkpoint_path, map_location=device)
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

    # Save middle slice for quick visual inspection
    middle_slice = sample[:, :, 32]

    output_path = output_dir / f"sample_epoch{epoch}_{i + 1}.png"

    plt.imshow(middle_slice, cmap="gray")
    plt.axis("off")
    plt.savefig(
        output_path,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()

    print(f"Saved sample {i + 1}/{num_samples}: {output_path}")


print("Sampling complete.")