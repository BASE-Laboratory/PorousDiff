"""
Deeper sample_diffusion.py

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

experiment_name = "ddpm_deeper_64"
checkpoint_epoch = 100
num_samples = 20
sample_shape = (1, 1, 64, 64, 64)

# Use this if training completed normally
checkpoint_path = Path(
    f"results/diffusion_checkpoints/{experiment_name}_epoch{checkpoint_epoch}.pth"
)

# If you want to load the final checkpoint instead, use this:
# checkpoint_path = Path(
#     f"results/diffusion_checkpoints/{experiment_name}_epoch{checkpoint_epoch}_final.pth"
# )

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
# Load improved/deeper DDPM model
# Must match train_diffusion_deeper.py exactly
# -----------------------------

model = DiffusionModelUNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    num_channels=(32, 64, 128),
    attention_levels=(False, False, True),
    num_res_blocks=2,
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
    noise = torch.randn(sample_shape).to(device)

    with torch.no_grad():
        sample = inferer.sample(
            input_noise=noise,
            diffusion_model=model,
            scheduler=scheduler,
        )

    sample = sample[0, 0].detach().cpu().numpy()
    middle_slice = sample[:, :, 32]

    output_path = output_dir / (
        f"{experiment_name}_epoch{checkpoint_epoch}_sample_{i + 1}.png"
    )

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