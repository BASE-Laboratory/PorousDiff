"""Sample a trained load-conditioned 3D DDPM across discrete states and continuous load sweeps.

Supports multi-sample statistical generation (`N >= 20` per load state), unseen intermediate
load interpolation, fixed-noise compression sweeps, DDIM/DDPM sampling, classifier-free
guidance, EMA weights, and OpenImpala 3D TIFF export.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from porousdiff.data import default_project_root
from porousdiff.metrics import evaluate_volume_physics, export_openimpala_tiff
from porousdiff.models import ContinuousLoadConditionedUNet, EMAModel, build_noise_scheduler
from porousdiff.preprocessing import parse_load_value


@torch.inference_mode()
def generate_batch(
    model: ContinuousLoadConditionedUNet,
    scheduler: object,
    initial_noise: torch.Tensor,
    load_value: float,
    guidance_scale: float,
    use_amp: bool,
    amp_dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run reverse diffusion from `initial_noise` `[B, 1, D, H, W]` at scalar `load_value`."""
    device = initial_noise.device
    bsz = initial_noise.shape[0]
    image = initial_noise.clone()
    loads = torch.full((bsz,), float(load_value), device=device, dtype=torch.float32)

    for timestep in scheduler.timesteps:  # type: ignore[attr-defined]
        t_int = int(timestep)
        t_batch = torch.full((bsz,), t_int, device=device, dtype=torch.long)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            pred = model.forward_cfg(
                image,
                t_batch,
                loads,
                guidance_scale=guidance_scale,
            )
        image = scheduler.step(pred.float(), t_int, image, generator=generator)[0]  # type: ignore[attr-defined]

    continuous = image[:, 0].float().cpu().numpy()
    normalized = np.clip((continuous + 1.0) * 0.5, 0.0, 1.0).astype(np.float32)
    binary = (normalized >= 0.5).astype(np.uint8)
    return binary, normalized


def save_orthoview_figure(binary: np.ndarray, title: str, png_path: Path, pdf_path: Path | None = None) -> None:
    mid_z, mid_y, mid_x = (s // 2 for s in binary.shape)
    views = [
        (binary[mid_z, :, :], "XY (In-Plane Sheet)"),
        (binary[:, mid_y, :], "XZ (Through-Plane Compression)"),
        (binary[:, :, mid_x], "YZ (Through-Plane Compression)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (view, vtitle) in zip(axes, views):
        ax.imshow(view, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(vtitle)
        ax.axis("off")
    fig.suptitle(f"{title} (solid frac = {binary.mean():.3f})")
    fig.tight_layout()
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    if pdf_path is not None:
        fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def save_sweep_panel(
    sweep_binaries: list[tuple[float, np.ndarray]],
    png_path: Path,
    pdf_path: Path,
) -> None:
    n_cols = len(sweep_binaries)
    fig, axes = plt.subplots(2, n_cols, figsize=(3.2 * n_cols, 6.2))
    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    for col, (load_mv, vol) in enumerate(sweep_binaries):
        mz, my, _ = (s // 2 for s in vol.shape)
        axes[0, col].imshow(vol[mz], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[0, col].set_title(f"{load_mv:g} mV/V\nφ_s = {vol.mean():.3f} (XY)")
        axes[0, col].axis("off")

        axes[1, col].imshow(vol[:, my, :], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[1, col].set_title("XZ (Compression ↓)")
        axes[1, col].axis("off")

    fig.suptitle("Virtual Compression Sweep (Fixed Initial Gaussian Noise Seed)", fontsize=13)
    fig.tight_layout()
    fig.savefig(png_path, dpi=260, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    proj_root = default_project_root()
    parser = argparse.ArgumentParser(
        description="Sample load-conditioned 3D microstructures across discrete and interpolated loads"
    )
    parser.add_argument("--experiment", default="carbon_conditional_96_v2")
    parser.add_argument("--output-root", type=Path, default=proj_root)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Explicit checkpoint path (defaults to best_val.pth if present, else latest.pth)",
    )
    parser.add_argument("--no-ema", action="store_true", help="Sample raw online weights instead of EMA")
    parser.add_argument("--sampler", choices=["ddim", "ddpm"], default="ddim")
    parser.add_argument("--steps", type=int, default=100, help="Inference denoising steps (e.g. 100 for DDIM, 1000 for DDPM)")
    parser.add_argument("--guidance-scale", type=float, default=1.5)
    parser.add_argument("--num-samples", type=int, default=20, help="Samples to generate per evaluated state")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--eval-states",
        nargs="+",
        default=None,
        help="States to sample (e.g. Op008mVV=0.008 Op023mVV=0.023); defaults to checkpoint states",
    )
    parser.add_argument(
        "--sweep-loads",
        default="0.008,0.015,0.023,0.033,0.043,0.053",
        help="Comma-separated load values for a fixed-noise continuous compression sweep (empty to skip)",
    )
    parser.add_argument("--export-openimpala", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.allow_cpu:
        device = torch.device("cpu")
    else:
        raise RuntimeError("CUDA GPU required (pass --allow-cpu for local CPU testing).")

    use_amp = device.type == "cuda"
    amp_dtype = (
        torch.bfloat16
        if (device.type == "cuda" and torch.cuda.is_bf16_supported())
        else torch.float16
    )

    ckpt_dir = args.output_root / "checkpoints" / args.experiment
    if args.checkpoint is not None:
        ckpt_path = args.checkpoint
    elif (ckpt_dir / "best_val.pth").is_file():
        ckpt_path = ckpt_dir / "best_val.pth"
    else:
        ckpt_path = ckpt_dir / "latest.pth"

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = saved["config"]
    patch_size = int(cfg["patch_size"])

    model = ContinuousLoadConditionedUNet(
        channels=cfg["channels"],
        attention_levels=cfg["attention_levels"],
        num_res_blocks=int(cfg["num_res_blocks"]),
        norm_num_groups=int(cfg["norm_groups"]),
        load_min=float(cfg["load_min"]),
        load_max=float(cfg["load_max"]),
        use_flash_attention=(device.type == "cuda"),
    ).to(device)

    if not args.no_ema and "ema" in saved:
        ema = EMAModel(model)
        ema.load_state_dict(saved["ema"])
        active_model = ema.shadow_model.to(device).eval()
        weight_source = "EMA"
    else:
        model.load_state_dict(saved["model"])
        active_model = model.eval()
        weight_source = "online"

    scheduler = build_noise_scheduler(
        sampler=args.sampler,
        num_train_timesteps=int(cfg["timesteps"]),
        schedule=str(cfg["schedule"]),
        prediction_type=str(cfg["prediction_type"]),
    )
    scheduler.set_timesteps(args.steps, device=device)

    print(
        f"Loaded {weight_source} weights from {ckpt_path} (epoch {int(saved['epoch']) + 1}) | "
        f"sampler={args.sampler} ({args.steps} steps, CFG={args.guidance_scale})",
        flush=True,
    )

    # Determine states to sample
    if args.eval_states:
        eval_states: dict[str, float] = {}
        for entry in args.eval_states:
            if "=" in entry:
                k, v = entry.split("=", 1)
                eval_states[k.strip()] = float(v.strip())
            else:
                eval_states[entry.strip()] = parse_load_value(entry.strip())
    else:
        eval_states = {str(k): float(v) for k, v in cfg["states"].items()}

    out_dir = args.output_root / "results" / args.experiment / "samples"
    figures_dir = args.output_root / "results" / args.experiment / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    per_sample_records: list[dict[str, object]] = []
    summary_records: list[dict[str, object]] = []

    for state_name, load_val in eval_states.items():
        state_dir = out_dir / state_name
        state_dir.mkdir(parents=True, exist_ok=True)

        gen = torch.Generator(device=device).manual_seed(args.seed)
        collected_bin: list[np.ndarray] = []
        started = time.perf_counter()

        remaining = args.num_samples
        sample_idx = 0
        while remaining > 0:
            cur_b = min(args.batch_size, remaining)
            noise = torch.randn(
                (cur_b, 1, patch_size, patch_size, patch_size),
                generator=gen,
                device=device,
            )
            binaries, probs = generate_batch(
                model=active_model,
                scheduler=scheduler,
                initial_noise=noise,
                load_value=load_val,
                guidance_scale=args.guidance_scale,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                generator=gen,
            )
            for b_vol, p_vol in zip(binaries, probs):
                sample_idx += 1
                collected_bin.append(b_vol)
                np.save(state_dir / f"sample_{sample_idx:03d}_binary.npy", b_vol)
                np.save(state_dir / f"sample_{sample_idx:03d}_prob.npy", p_vol)

                if args.export_openimpala:
                    export_openimpala_tiff(
                        b_vol, state_dir / "openimpala" / f"sample_{sample_idx:03d}.tif"
                    )

                phys = evaluate_volume_physics(b_vol)
                rec: dict[str, object] = {
                    "state": state_name,
                    "load_mv_per_v": load_val,
                    "sample_id": sample_idx,
                }
                rec.update({k: round(float(v), 6) if np.isfinite(v) else float("nan") for k, v in phys.items()})
                per_sample_records.append(rec)

                if sample_idx == 1:
                    save_orthoview_figure(
                        b_vol,
                        title=f"{state_name} ({load_val:g} mV/V)",
                        png_path=figures_dir / f"figure_generated_{state_name}.png",
                        pdf_path=figures_dir / f"figure_generated_{state_name}.pdf",
                    )
            remaining -= cur_b

        elapsed = time.perf_counter() - started
        stacked_state = np.stack(collected_bin, axis=0)
        np.save(state_dir / "generated_samples.npy", stacked_state)

        sfs = [float(v.mean()) for v in collected_bin]
        summary_row = {
            "state": state_name,
            "load_mv_per_v": load_val,
            "num_samples": args.num_samples,
            "solid_fraction_mean": round(float(np.mean(sfs)), 6),
            "solid_fraction_std": round(float(np.std(sfs)), 6),
            "solid_fraction_min": round(float(np.min(sfs)), 6),
            "solid_fraction_max": round(float(np.max(sfs)), 6),
            "sampling_seconds": round(elapsed, 2),
        }
        summary_records.append(summary_row)
        print(json.dumps(summary_row), flush=True)

    # Save CSVs
    if per_sample_records:
        with (out_dir / "per_sample_physics.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_sample_records[0].keys()))
            writer.writeheader()
            writer.writerows(per_sample_records)

    if summary_records:
        with (out_dir / "sampling_metrics.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_records[0].keys()))
            writer.writeheader()
            writer.writerows(summary_records)

    # Optional continuous fixed-noise compression sweep
    if args.sweep_loads.strip():
        sweep_vals = [float(x.strip()) for x in args.sweep_loads.split(",") if x.strip()]
        sweep_gen = torch.Generator(device=device).manual_seed(args.seed + 777)
        shared_noise = torch.randn(
            (1, 1, patch_size, patch_size, patch_size),
            generator=sweep_gen,
            device=device,
        )
        sweep_dir = out_dir / "continuous_sweep"
        sweep_dir.mkdir(parents=True, exist_ok=True)

        sweep_pairs: list[tuple[float, np.ndarray]] = []
        for mv in sweep_vals:
            step_gen = torch.Generator(device=device).manual_seed(args.seed + 888)
            b_vol, _ = generate_batch(
                model=active_model,
                scheduler=scheduler,
                initial_noise=shared_noise,
                load_value=mv,
                guidance_scale=args.guidance_scale,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                generator=step_gen,
            )
            vol = b_vol[0]
            tag = f"{mv:.4f}".replace(".", "p")
            np.save(sweep_dir / f"sweep_{tag}mVV.npy", vol)
            sweep_pairs.append((mv, vol))

        save_sweep_panel(
            sweep_pairs,
            png_path=figures_dir / "compression_sweep_orthoviews.png",
            pdf_path=figures_dir / "compression_sweep_orthoviews.pdf",
        )
        print(f"Saved continuous compression sweep ({len(sweep_pairs)} loads) -> {figures_dir}", flush=True)


if __name__ == "__main__":
    main()
