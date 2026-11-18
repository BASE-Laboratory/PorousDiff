import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import lpips

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("CUDA available:", torch.cuda.is_available())
print("Using device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# Paths
project_dir = Path.home() / "PorousDiff"
data_path = project_dir / "data/processed/patches64"
results_dir = project_dir / "results"

recon_dir = results_dir / "reconstructions"
loss_dir = results_dir / "loss_curves"
ckpt_dir = results_dir / "checkpoints"
log_dir = results_dir / "logs"

for d in [recon_dir, loss_dir, ckpt_dir, log_dir]:
    d.mkdir(parents=True, exist_ok=True)

# Load patches
files = sorted(data_path.glob("*.npy"))
print("Total patches:", len(files))

data = [np.load(f) for f in files]
data = np.stack(data)

data = torch.tensor(data, dtype=torch.float32).unsqueeze(1)
print("Data shape:", data.shape)


class AE(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv3d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv3d(32, 64, 3, stride=2, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(16, 1, kernel_size=3, padding=1),

            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


model = AE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

mse_fn = nn.MSELoss()
lpips_fn = lpips.LPIPS(net="alex").to(device)
lpips_weight = 0.1

num_epochs = 25
epoch_tag = f"epoch{num_epochs}"
dataset_tag = "1200patches_800slices_lpips"

total_losses = []
mse_losses = []
lpips_losses = []

# Training loop
for epoch in range(num_epochs):
    total_epoch_loss = 0.0
    total_epoch_mse = 0.0
    total_epoch_lpips = 0.0

    for x in data:
        x = x.unsqueeze(0).to(device)

        out = model(x)

        mse_loss = mse_fn(out, x)

        # Middle 2D slice from 3D volume: (B, C, D, H, W) -> (B, C, H, W)
        out_slice = out[:, :, 32, :, :]
        x_slice = x[:, :, 32, :, :]

        # Convert grayscale to 3-channel for LPIPS
        out_slice = out_slice.repeat(1, 3, 1, 1)
        x_slice = x_slice.repeat(1, 3, 1, 1)

        # LPIPS expects range [-1, 1]
        out_slice = out_slice * 2 - 1
        x_slice = x_slice * 2 - 1

        lpips_loss = lpips_fn(out_slice, x_slice).mean()

        loss = mse_loss + lpips_weight * lpips_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_epoch_loss += loss.item()
        total_epoch_mse += mse_loss.item()
        total_epoch_lpips += lpips_loss.item()

    avg_total = total_epoch_loss / len(data)
    avg_mse = total_epoch_mse / len(data)
    avg_lpips = total_epoch_lpips / len(data)

    total_losses.append(avg_total)
    mse_losses.append(avg_mse)
    lpips_losses.append(avg_lpips)

    print(
        f"Epoch {epoch}: "
        f"Total Loss = {avg_total:.6f}, "
        f"MSE = {avg_mse:.6f}, "
        f"LPIPS = {avg_lpips:.6f}"
    )

print("Training complete")

# Save checkpoint
torch.save(
    model.state_dict(),
    ckpt_dir / f"autoencoder_{epoch_tag}_{dataset_tag}.pth"
)
print("Saved checkpoint")

# Save loss curve
plt.figure(figsize=(7, 5))
plt.plot(total_losses, label="Total Loss")
plt.plot(mse_losses, label="MSE Loss")
plt.plot(lpips_losses, label="LPIPS Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve with LPIPS")
plt.legend()
plt.tight_layout()
plt.savefig(loss_dir / f"loss_curve_{epoch_tag}_{dataset_tag}.png", dpi=150)
plt.close()

print("Saved loss curve")

# Save reconstruction
model.eval()

with torch.no_grad():
    sample = data[0].unsqueeze(0).to(device)
    recon = model(sample)

orig_slice = sample.cpu().numpy()[0, 0, 32]
recon_slice = recon.cpu().numpy()[0, 0, 32]

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(orig_slice, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Reconstructed")
plt.imshow(recon_slice, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.savefig(
    recon_dir / f"reconstruction_{epoch_tag}_{dataset_tag}.png",
    dpi=150
)
plt.close()

print("Saved reconstruction figure")