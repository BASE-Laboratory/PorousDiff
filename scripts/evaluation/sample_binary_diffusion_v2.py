"""
PorousDiff — Op001 Binary DDPM v2 Sampling

Generate synthetic 64x64x64 binary XCT microstructures
using the trained 3D DDPM v2 checkpoint.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler
from generative.inferers import DiffusionInferer


ROOT = Path("/projects/u6uc/asif")
EXPERIMENT = "op001_binary_v2"

CHECKPOINT_DIR = ROOT / "checkpoints" / EXPERIMENT
OUTPUT_DIR = ROOT / "results" / EXPERIMENT / "samples"


def main():
    parser = argparse.ArgumentParser(
        description="Sample Op001 Binary DDPM v2"
    )
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.epoch < 1 or args.num_samples < 1:
        raise ValueError("Epoch and num-samples must be positive")

    if not torch.cuda.is_available():
        raise RuntimeError("GPU required. Run through Slurm.")

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print("GPU:", torch.cuda.get_device_name(0), flush=True)

    checkpoint_path = (
        CHECKPOINT_DIR / f"epoch_{args.epoch:03d}.pth"
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    # Same architecture as the v2 training script
    model = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        num_channels=(32, 64, 128),
        attention_levels=(False, False, True),
        num_res_blocks=1,
    ).to(device)

    # v2 checkpoints store weights inside a dictionary
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Loaded checkpoint:", checkpoint_path, flush=True)

    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        schedule="scaled_linear_beta",
    )

    inferer = DiffusionInferer(scheduler)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for i in range(args.num_samples):
        noise = torch.randn(
            (1, 1, 64, 64, 64),
            device=device,
        )

        print(
            f"Generating sample {i + 1}/{args.num_samples}",
            flush=True,
        )

        with torch.inference_mode():
            generated = inferer.sample(
                input_noise=noise,
                diffusion_model=model,
                scheduler=scheduler,
            )

        continuous = generated[0, 0].cpu().numpy()

        # Map approximately [-1, 1] to [0, 1]
        normalized = np.clip(
            (continuous + 1.0) / 2.0,
            0.0,
            1.0,
        ).astype(np.float32)

        # Convert generated volume to binary microstructure
        binary = (normalized >= 0.5).astype(np.uint8)

        stem = f"op001_binary_v2_epoch{args.epoch:03d}_sample{i+1:03d}"

        binary_path = OUTPUT_DIR / f"{stem}.npy"
        continuous_path = OUTPUT_DIR / f"{stem}_continuous.npy"
        figure_path = OUTPUT_DIR / f"{stem}_orthogonal.png"

        np.save(binary_path, binary)
        np.save(continuous_path, normalized)

        mid = 32

        views = [
            (binary[mid, :, :], "XY"),
            (binary[:, mid, :], "XZ"),
            (binary[:, :, mid], "YZ"),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        for ax, (view, title) in zip(axes, views):
            ax.imshow(
                view,
                cmap="gray",
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )
            ax.set_title(title)
            ax.axis("off")

        fig.suptitle(
            f"Op001 Binary DDPM v2 — Epoch {args.epoch}"
        )
        fig.tight_layout()
        fig.savefig(figure_path, dpi=180)
        plt.close(fig)

        print("Saved:", binary_path, flush=True)
        print("Saved:", figure_path, flush=True)
        print(
            f"Generated white fraction: {binary.mean():.6f}",
            flush=True,
        )

    print("Sampling completed successfully.", flush=True)


if __name__ == "__main__":
    main()
