"""
train_diffusion_op026.py

Train a baseline 3D DDPM on Op026mVV TiO2 64x64x64 patches.

Project: PorousDiff
Code: Asif Ali
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
    Loads preprocessed 64x64x64 NumPy patches.
    Each patch is returned as [1, 64, 64, 64].
    """

    def __init__(self, patch_dir):
        self.patch_files = sorted(list(Path(patch_dir).glob("*.npy")))

    def __len__(self):
        return len(self.patch_files)

    def __getitem__(self, idx):
        patch = np.load(self.patch_files[idx]).astype(np.float32)

        # Normalize only if patch is not already in [0, 1]
        if patch.max() > 1:
            patch = patch / patch.max()

        patch = torch.tensor(patch).float()

        # Add channel dimension
        patch = patch.unsqueeze(0)

        return patch


# -----------------------------
# Configuration
# -----------------------------

patch_dir = "data/processed/patches64_op026"
experiment_name = "ddpm_op026_64"

batch_size = 1
num_epochs = 100
learning_rate = 1e-4
save_every = 10

checkpoint_dir = Path("results/diffusion_checkpoints")
loss_curve_dir = Path("results/loss_curves")

checkpoint_dir.mkdir(parents=True, exist_ok=True)
loss_curve_dir.mkdir(parents=True, exist_ok=True)


# -----------------------------
# RESUME CHANGE
# -----------------------------
# Set resume_training = True only when continuing from an existing checkpoint.
# For a fresh run, set resume_training = False.
#
# Current use case:
# The previous Op026 run stopped at epoch 75 because the SLURM time limit ended.
# The latest safe checkpoint saved was epoch 70.
# So we resume from epoch 70 and continue training to epoch 100.

resume_training = True
resume_epoch = 70
resume_checkpoint = checkpoint_dir / f"{experiment_name}_epoch{resume_epoch}.pth"

# For a fresh run in the future, use:
# resume_training = False
# resume_epoch = 0


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
    num_workers=0,
)


# -----------------------------
# Baseline 3D DDPM model
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
# RESUME CHANGE
# -----------------------------
# Load previously saved model weights if resuming training.
# IMPORTANT:
# This resumes the model weights only.
# It does NOT restore optimizer state.
# That is acceptable for this quick continuation experiment,
# but future production runs should save optimizer state too.

if resume_training:
    model.load_state_dict(
        torch.load(resume_checkpoint, map_location=device)
    )

    print(f"Resumed model weights from: {resume_checkpoint}")
    start_epoch = resume_epoch
else:
    print("Starting fresh training run.")
    start_epoch = 0


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
# Training loop
# -----------------------------

model.train()
epoch_losses = []

# RESUME CHANGE:
# range(start_epoch, num_epochs) means:
# - if start_epoch = 70, training continues from epoch 71 to 100
# - if start_epoch = 0, training starts from epoch 1
for epoch in range(start_epoch, num_epochs):
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

    # Save/update loss curve after each epoch
    loss_curve_path = loss_curve_dir / f"{experiment_name}_resume_from{resume_epoch}_to{num_epochs}_loss.png"

    plt.figure()
    plt.plot(epoch_losses)
    plt.xlabel("Epoch after resume")
    plt.ylabel("Average MSE Loss")
    plt.title("Op026 3D DDPM Training Loss")
    plt.savefig(loss_curve_path)
    plt.close()

    # Save checkpoint every 10 epochs
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