"""
Sample a trained conditional 3D DDPM at 8, 18 and 33 mV.
Save continuous and binary volumes, figures and sampling metrics.
"""

import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler

ROOT = Path("/projects/u6uc/asif")
EXPERIMENT = "carbon_conditional_96_v1"
CHECKPOINT = ROOT / "checkpoints" / EXPERIMENT / "latest.pth"
OUTPUT = ROOT / "results" / EXPERIMENT / "samples"
FIGURES = ROOT / "paper" / "figures"

STATES = {
    "Op008mVV": 8.0,
    "Op018mVV": 18.0,
    "Op033mVV": 33.0,
}


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required; use Slurm.")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

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

    condition_encoder = torch.nn.Sequential(
        torch.nn.Linear(1, 32),
        torch.nn.SiLU(),
        torch.nn.Linear(32, 32),
    ).to(device)

    checkpoint = torch.load(
        CHECKPOINT, map_location="cpu", weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    condition_encoder.load_state_dict(
        checkpoint["condition_encoder"]
    )

    model.eval()
    condition_encoder.eval()

    print("Checkpoint:", CHECKPOINT, flush=True)
    print("Completed epochs:", checkpoint["epoch"] + 1, flush=True)
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        schedule="scaled_linear_beta",
    )

    # Identical initial noise for every condition.
    initial_noise = torch.randn(
        (1, 1, 96, 96, 96), device=device
    )

    results = []

    for name, mv in STATES.items():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()

        condition = torch.tensor(
            [[(mv - 20.5) / 12.5]],
            device=device,
            dtype=torch.float32,
        )

        with torch.inference_mode():
            context = condition_encoder(condition).unsqueeze(1)
            image = initial_noise.clone()

            for timestep in scheduler.timesteps:
                t = torch.as_tensor(
                    [int(timestep)], device=device
                ).long()

                predicted_noise = model(
                    image, t, context=context
                )

                image = scheduler.step(
                    predicted_noise, int(timestep), image
                )[0]

        torch.cuda.synchronize()
        seconds = time.perf_counter() - started

        continuous = image[0, 0].float().cpu().numpy()
        normalized = np.clip(
            (continuous + 1.0) / 2.0, 0, 1
        ).astype(np.float32)
        binary = (normalized >= 0.5).astype(np.uint8)

        np.save(OUTPUT / f"{name}_generated_binary.npy", binary)
        np.save(
            OUTPUT / f"{name}_generated_continuous.npy",
            normalized,
        )

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        views = [
            (binary[48], "XY"),
            (binary[:, 48, :], "XZ"),
            (binary[:, :, 48], "YZ"),
        ]

        for ax, (view, title) in zip(axes, views):
            ax.imshow(
                view, cmap="gray", vmin=0, vmax=1,
                interpolation="nearest"
            )
            ax.set_title(title)
            ax.axis("off")

        fig.suptitle(f"Conditional DDPM — {mv:g} mV")
        fig.tight_layout()
        fig.savefig(
            FIGURES / f"figure_generated_{name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        fig.savefig(
            FIGURES / f"figure_generated_{name}.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)

        record = {
            "state": name,
            "load_cell_mv": mv,
            "solid_fraction": float(binary.mean()),
            "sampling_seconds": round(seconds, 2),
            "peak_allocated_gib": round(
                torch.cuda.max_memory_allocated() / 1024**3, 3
            ),
            "peak_reserved_gib": round(
                torch.cuda.max_memory_reserved() / 1024**3, 3
            ),
        }
        results.append(record)
        print(record, flush=True)

    with (OUTPUT / "sampling_metrics.csv").open(
        "w", newline=""
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=results[0].keys()
        )
        writer.writeheader()
        writer.writerows(results)

    print("Conditional sampling completed.", flush=True)


if __name__ == "__main__":
    main()
