"""
train_diffusion.py

Train a 3D Denoising Diffusion Probabilistic Model (DDPM)
on 64x64x64 battery electrode tomography patches.

Project: PorousDiff
code: Asif Ali
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler


class PatchDataset(Dataset):
    """
    PyTorch Dataset for loading preprocessed 64x64x64 NumPy patches.
    Each patch is returned with shape [1, 64, 64, 64].
    """

    def __init__(self, patch_dir):
        self.patch_files = sorted(list(Path(patch_dir).glob("*.npy")))

    def __len__(self):
        return len(self.patch_files)

    def __getitem__(self, idx):
        # Load one 3D patch
        patch = np.load(self.patch_files[idx]).astype(np.float32)

        # Normalize intensity values to [0, 1] if needed
        if patch.max() > 1:
            patch = patch / patch.max()

        # Convert to tensor
        patch = torch.tensor(patch).float()

        # Add channel dimension: [64,64,64] -> [1,64,64,64]
        patch = patch.unsqueeze(0)

        return patch


# -----------------------------
# Configuration
# -----------------------------

patch_dir = "data/processed/patches64"
batch_size = 1
num_epochs = 100
learning_rate = 1e-4

checkpoint_dir = Path("results/diffusion_checkpoints")
loss_curve_dir = Path("results/loss_curves")

checkpoint_dir.mkdir(parents=True, exist_ok=True)
loss_curve_dir.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Device setup
# -----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# -----------------------------
# Dataset and DataLoader
# -----------------------------

dataset = PatchDataset(patch_dir)

print(f"Found {len(dataset)} patches")

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,  # safer on HPC interactive sessions
)


# -----------------------------
# 3D DDPM model
# -----------------------------

model = DiffusionModelUNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    num_channels=(32, 64, 128),
    attention_levels=(False, False, True),
    num_res_blocks=1,
).to(device)


# -----------------------------
# DDPM noise scheduler
# -----------------------------

scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    schedule="scaled_linear_beta",
)


# -----------------------------
# Optimizer
# -----------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
)


# -----------------------------
# Training loop
# -----------------------------

model.train()
epoch_losses = []

for epoch in range(num_epochs):
    epoch_loss = 0.0

    for step, clean_images in enumerate(dataloader):
        clean_images = clean_images.to(device)

        # Sample Gaussian noise
        noise = torch.randn_like(clean_images).to(device)

        # Random diffusion timestep for each batch item
        timesteps = torch.randint(
            0,
            scheduler.num_train_timesteps,
            (clean_images.shape[0],),
            device=device,
        ).long()

        # Add noise to clean microstructure patches
        noisy_images = scheduler.add_noise(
            original_samples=clean_images,
            noise=noise,
            timesteps=timesteps,
        )

        # Model predicts the noise added to the image
        noise_pred = model(noisy_images, timesteps)

        # DDPM objective: predict noise using MSE loss
        loss = F.mse_loss(
            noise_pred.float(),
            noise.float(),
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if step % 10 == 0:
            print(
                f"Epoch {epoch + 1} "
                f"Step {step} "
                f"Loss: {loss.item():.6f}"
            )

    avg_loss = epoch_loss / len(dataloader)
    epoch_losses.append(avg_loss)

    print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.6f}")

    # Save/update loss curve after each epoch
    plt.figure()
    plt.plot(epoch_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Average MSE Loss")
    plt.title("3D DDPM Training Loss")
    plt.savefig(loss_curve_dir / f"ddpm_3d_epoch{num_epochs}_loss.png")
    plt.close()


# -----------------------------
# Save final checkpoint
# -----------------------------

checkpoint_path = checkpoint_dir / f"ddpm_3d_epoch{num_epochs}.pth"

torch.save(
    model.state_dict(),
    checkpoint_path,
)

print("Training complete.")
print(f"Saved checkpoint: {checkpoint_path}")
print(f"Saved loss curve: {loss_curve_dir / f'ddpm_3d_epoch{num_epochs}_loss.png'}")