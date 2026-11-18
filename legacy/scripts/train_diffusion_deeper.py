"""
Deeper train_diffusion.py

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
    def __init__(self, patch_dir):
        self.patch_files = sorted(list(Path(patch_dir).glob("*.npy")))

    def __len__(self):
        return len(self.patch_files)

    def __getitem__(self, idx):
        patch = np.load(self.patch_files[idx]).astype(np.float32)

        if patch.max() > 1:
            patch = patch / patch.max()

        patch = torch.tensor(patch).float()
        patch = patch.unsqueeze(0)

        return patch


# -----------------------------
# Configuration
# -----------------------------

patch_dir = "data/processed/patches64"
experiment_name = "ddpm_deeper_64"

batch_size = 1
num_epochs = 100
learning_rate = 1e-4
save_every = 10

checkpoint_dir = Path("results/diffusion_checkpoints")
loss_curve_dir = Path("results/loss_curves")

checkpoint_dir.mkdir(parents=True, exist_ok=True)
loss_curve_dir.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Device
# -----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# -----------------------------
# Data
# -----------------------------

dataset = PatchDataset(patch_dir)

print(f"Found {len(dataset)} patches")

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
)


# -----------------------------
# Improved 3D DDPM model
# -----------------------------

model = DiffusionModelUNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    num_channels=(32, 64, 128),
    attention_levels=(False, False, True),
    num_res_blocks=2,
).to(device)


# -----------------------------
# DDPM scheduler
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
# Training
# -----------------------------

model.train()
epoch_losses = []

for epoch in range(num_epochs):
    epoch_loss = 0.0

    for step, clean_images in enumerate(dataloader):
        clean_images = clean_images.to(device)

        noise = torch.randn_like(clean_images).to(device)

        timesteps = torch.randint(
            0,
            scheduler.num_train_timesteps,
            (clean_images.shape[0],),
            device=device,
        ).long()

        noisy_images = scheduler.add_noise(
            original_samples=clean_images,
            noise=noise,
            timesteps=timesteps,
        )

        noise_pred = model(noisy_images, timesteps)

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

    # Save/update loss curve after every epoch
    loss_curve_path = loss_curve_dir / f"{experiment_name}_epoch{num_epochs}_loss.png"

    plt.figure()
    plt.plot(epoch_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Average MSE Loss")
    plt.title("Improved 3D DDPM Training Loss")
    plt.savefig(loss_curve_path)
    plt.close()

    # Save checkpoint every N epochs
    if (epoch + 1) % save_every == 0:
        checkpoint_path = checkpoint_dir / f"{experiment_name}_epoch{epoch + 1}.pth"

        torch.save(
            model.state_dict(),
            checkpoint_path,
        )

        print(f"Saved checkpoint: {checkpoint_path}")


# -----------------------------
# Save final checkpoint
# -----------------------------

final_checkpoint_path = checkpoint_dir / f"{experiment_name}_epoch{num_epochs}_final.pth"

torch.save(
    model.state_dict(),
    final_checkpoint_path,
)

print("Training complete.")
print(f"Saved final checkpoint: {final_checkpoint_path}")
print(f"Saved loss curve: {loss_curve_path}")