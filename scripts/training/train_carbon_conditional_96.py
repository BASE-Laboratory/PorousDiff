
"""
PorousDiff — Carbon-felt conditional 3D DDPM (96³).

One shared model conditioned on load-cell readings: 8, 18, 33 mV.
Records losses, epoch timings, GPU memory, and resumable checkpoints.
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler

ROOT = Path("/projects/u6uc/asif")
DATA_ROOT = ROOT / "data/carbon_virgin"
EXPERIMENT = "carbon_conditional_96_v1"
RESULT_DIR = ROOT / "results" / EXPERIMENT
CHECKPOINT_DIR = ROOT / "checkpoints" / EXPERIMENT

STATES = {"Op008mVV": 8.0, "Op018mVV": 18.0, "Op033mVV": 33.0}
VERSION = "carbon_sigma50_v1"


class CarbonDataset(Dataset):
    def __init__(self, split):
        self.samples = []
        for state, mv in STATES.items():
            folder = DATA_ROOT / (
                f"{state}_binary_patches96_{VERSION}"
            ) / split
            files = sorted(folder.glob("*.npy"))
            if not files:
                raise RuntimeError(f"No patches in {folder}")
            self.samples.extend((p, mv) for p in files)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, mv = self.samples[index]
        patch = np.load(path)
        if patch.shape != (96, 96, 96) or patch.dtype != np.uint8:
            raise ValueError(f"Invalid patch: {path}")
        image = torch.from_numpy(
            patch.astype(np.float32)
        ).unsqueeze(0) * 2 - 1

        # Normalize the sensor reading to approximately [-1, 1].
        condition = torch.tensor(
            [(mv - 20.5) / 12.5], dtype=torch.float32
        )
        return image, condition


def checkpoint(path, model, optimizer, epoch, history):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "history": history,
        "rng_torch": torch.get_rng_state(),
        "rng_numpy": np.random.get_state(),
        "rng_python": random.getstate(),
    }, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Limit training batches per epoch for smoke tests")
    parser.add_argument("--max-val-steps", type=int, default=0)
    parser.add_argument(
        "--experiment",
        default="carbon_conditional_96_v1",
        help="Separate experiment name for checkpoints and results",
    )
    args = parser.parse_args()

    if not args.experiment.replace("_", "").isalnum():
        parser.error("Invalid experiment name")

    global RESULT_DIR, CHECKPOINT_DIR
    RESULT_DIR = ROOT / "results" / args.experiment
    CHECKPOINT_DIR = ROOT / "checkpoints" / args.experiment

    if not torch.cuda.is_available():
        raise RuntimeError("GPU required; submit through Slurm")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

    train_data = CarbonDataset("train")
    val_data = CarbonDataset("val")
    print("Training patches:", len(train_data), flush=True)
    print("Validation patches:", len(val_data), flush=True)

    train_loader = DataLoader(
        train_data, batch_size=args.batch_size,
        shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size,
        shuffle=False, num_workers=2, pin_memory=True
    )

    model = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        num_channels=(32, 64, 128),
        attention_levels=(False, False, True),
        num_res_blocks=1,
        with_conditioning=True,
        cross_attention_dim=32,
    ).to(device)

    # Embed the scalar load-cell reading into cross-attention context.
    condition_encoder = torch.nn.Sequential(
        torch.nn.Linear(1, 32),
        torch.nn.SiLU(),
        torch.nn.Linear(32, 32),
    ).to(device)

    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        schedule="scaled_linear_beta",
    )

    optimizer = torch.optim.AdamW(
        list(model.parameters()) +
        list(condition_encoder.parameters()),
        lr=args.lr,
    )

    def loss_fn(batch):
        images, conditions = batch
        images = images.to(device, non_blocking=True)
        conditions = conditions.to(device, non_blocking=True)
        context = condition_encoder(conditions).unsqueeze(1)

        noise = torch.randn_like(images)
        t = torch.randint(
            0, 1000, (images.shape[0],), device=device
        ).long()
        noisy = scheduler.add_noise(images, noise, t)
        predicted = model(noisy, t, context=context)
        return F.mse_loss(predicted.float(), noise.float())

    latest = CHECKPOINT_DIR / "latest.pth"
    history = []
    start_epoch = 0

    if args.resume:
        saved = torch.load(latest, map_location=device,
                           weights_only=False)
        model.load_state_dict(saved["model"])
        condition_encoder.load_state_dict(saved["condition_encoder"])
        optimizer.load_state_dict(saved["optimizer"])
        history = saved["history"]
        start_epoch = saved["epoch"] + 1
        torch.set_rng_state(saved["rng_torch"].cpu())
        np.random.set_state(saved["rng_numpy"])
        random.setstate(saved["rng_python"])
    elif latest.exists():
        raise FileExistsError(
            f"{latest} exists; use --resume"
        )

    for epoch in range(start_epoch, args.epochs):
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        model.train()
        condition_encoder.train()
        train_total = 0.0
        train_steps = 0

        for images, conditions in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn((images, conditions))
            loss.backward()
            optimizer.step()
            train_total += loss.item()
            train_steps += 1
            if train_steps % 25 == 0:
                print(f"Epoch {epoch+1} step {train_steps}: "
                      f"loss={loss.item():.6f}", flush=True)
            if args.max_steps and train_steps >= args.max_steps:
                break

        model.eval()
        condition_encoder.eval()
        val_total = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch in val_loader:
                val_total += loss_fn(batch).item()
                val_steps += 1
                if args.max_val_steps and val_steps >= args.max_val_steps:
                    break

        torch.cuda.synchronize()
        record = {
            "epoch": epoch + 1,
            "train_loss": train_total / train_steps,
            "val_loss": val_total / val_steps,
            "epoch_seconds": round(
                time.perf_counter() - started, 2
            ),
            "peak_allocated_gib": round(
                torch.cuda.max_memory_allocated() / 1024**3, 3
            ),
            "peak_reserved_gib": round(
                torch.cuda.max_memory_reserved() / 1024**3, 3
            ),
        }
        history.append(record)
        print(json.dumps(record), flush=True)

        with (RESULT_DIR / "losses.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=record.keys())
            writer.writeheader()
            writer.writerows(history)

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "condition_encoder": condition_encoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "history": history,
            "rng_torch": torch.get_rng_state(),
            "rng_numpy": np.random.get_state(),
            "rng_python": random.getstate(),
        }
        temporary = latest.with_suffix(".pth.tmp")
        torch.save(state, temporary)
        temporary.replace(latest)
        print("Checkpoint:", latest, flush=True)

    print("Training completed.", flush=True)


if __name__ == "__main__":
    main()
