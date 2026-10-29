
"""
PorousDiff — Op001 Binary DDPM v2

Train a 3D DDPM on binary 64x64x64 XCT patches.
Includes validation, reproducibility, checkpointing and resume support.
"""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler


ROOT = Path("/projects/u6uc/asif")
DATA_DIR = ROOT / "data/Op001_binary_patches96_v2"
EXPERIMENT = "op001_binary_96_v2"

CHECKPOINT_DIR = ROOT / "checkpoints" / EXPERIMENT
RESULT_DIR = ROOT / "results" / EXPERIMENT


class BinaryPatchDataset(Dataset):
    def __init__(self, directory):
        self.files = sorted(Path(directory).glob("*.npy"))
        if not self.files:
            raise RuntimeError(f"No patches found in {directory}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        patch = np.load(self.files[index])

        if patch.shape != (96, 96, 96):
            raise ValueError(f"Invalid patch shape: {self.files[index]}")
        if patch.dtype != np.uint8:
            raise ValueError(f"Invalid dtype: {self.files[index]}")

        patch = torch.from_numpy(
            patch.astype(np.float32)
        ).unsqueeze(0)

        return patch * 2.0 - 1.0


def save_checkpoint(path, model, optimizer, epoch, losses):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "losses": losses,
            "rng_torch": torch.get_rng_state(),
            "rng_numpy": np.random.get_state(),
            "rng_python": random.getstate(),
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Epochs and batch size must be positive")

    if not torch.cuda.is_available():
        raise RuntimeError("GPU required. Submit using Slurm.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    train_dataset = BinaryPatchDataset(DATA_DIR / "train")
    val_dataset = BinaryPatchDataset(DATA_DIR / "val")

    print("Training patches:", len(train_dataset), flush=True)
    print("Validation patches:", len(val_dataset), flush=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        num_channels=(32, 64, 128),
        attention_levels=(False, False, True),
        num_res_blocks=1,
    ).to(device)

    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        schedule="scaled_linear_beta",
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
    )

    latest_checkpoint = CHECKPOINT_DIR / "latest.pth"
    start_epoch = 0
    losses = []

    if args.resume:
        if not latest_checkpoint.exists():
            raise FileNotFoundError(latest_checkpoint)

        checkpoint = torch.load(
            latest_checkpoint,
            map_location=device,
            weights_only=False,
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )
        start_epoch = checkpoint["epoch"] + 1
        losses = checkpoint["losses"]

        torch.set_rng_state(checkpoint["rng_torch"].cpu())
        np.random.set_state(checkpoint["rng_numpy"])
        random.setstate(checkpoint["rng_python"])

        print(
            f"Resuming from epoch {start_epoch + 1}",
            flush=True,
        )
    elif latest_checkpoint.exists():
        raise FileExistsError(
            f"{latest_checkpoint} already exists. "
            "Use --resume or a new experiment."
        )

    def diffusion_loss(images):
        images = images.to(device, non_blocking=True)
        noise = torch.randn_like(images)

        timesteps = torch.randint(
            0,
            scheduler.num_train_timesteps,
            (images.shape[0],),
            device=device,
        ).long()

        noisy = scheduler.add_noise(
            original_samples=images,
            noise=noise,
            timesteps=timesteps,
        )

        predicted_noise = model(noisy, timesteps)

        return F.mse_loss(
            predicted_noise.float(),
            noise.float(),
        )

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_total = 0.0

        for step, images in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)

            loss = diffusion_loss(images)
            loss.backward()
            optimizer.step()

            train_total += loss.item()

            if step % 100 == 0:
                print(
                    f"Epoch {epoch + 1}/{args.epochs} "
                    f"Step {step}/{len(train_loader)} "
                    f"Loss {loss.item():.6f}",
                    flush=True,
                )

        train_loss = train_total / len(train_loader)

        model.eval()
        val_total = 0.0

        with torch.no_grad():
            for images in val_loader:
                val_loss = diffusion_loss(images)
                val_total += val_loss.item()

        val_loss = val_total / len(val_loader)

        losses.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        print(
            f"Epoch {epoch + 1} complete | "
            f"Train: {train_loss:.6f} | "
            f"Val: {val_loss:.6f}",
            flush=True,
        )

        epoch_checkpoint = (
            CHECKPOINT_DIR / f"epoch_{epoch + 1:03d}.pth"
        )

        save_checkpoint(
            epoch_checkpoint, model, optimizer, epoch, losses
        )
        save_checkpoint(
            latest_checkpoint, model, optimizer, epoch, losses
        )

        with (RESULT_DIR / "losses.csv").open(
            "w", newline=""
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["epoch", "train_loss", "val_loss"],
            )
            writer.writeheader()
            writer.writerows(losses)

        print(
            f"Checkpoint saved: {epoch_checkpoint}",
            flush=True,
        )

    print("Training completed.", flush=True)


if __name__ == "__main__":
    main()
