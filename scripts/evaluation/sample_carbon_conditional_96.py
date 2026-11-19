"""Sample trained PorousDiff 3D models across single-material or multi-electrode 3-phase states.

Supports:
  - 2-phase (`{0, 1}`) and hierarchical 3-phase (`{0: pore, 1: carbon, 2: coating}`) decoding,
  - Multi-electrode (`--manifest-json` or `--material`) + voxel-size (`--voxel-size-um`) conditioning,
  - Unseen held-out load interpolation, fixed-noise compression sweeps, DDIM/DDPM, CFG, EMA,
    and OpenImpala multi-page TIFF export.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from matplotlib.colors import ListedColormap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from porousdiff.data import (
    decode_channels_to_phases,
    default_project_root,
    resolve_checkpoint_path,
    resolve_manifest_state_subset,
    resolve_material_id,
    resolve_scan_load_value,
)
from porousdiff.metrics import (
    evaluate_multiphase_physics,
    evaluate_volume_physics,
    export_openimpala_tiff,
)
from porousdiff.models import (
    ContinuousLoadConditionedUNet,
    EMAModel,
    build_noise_scheduler,
    infer_use_ordinal_material_channel,
)
from porousdiff.preprocessing import infer_material_name, parse_load_value
from porousdiff.transport_and_xrd import (
    has_openimpala_python,
    run_openimpala_rev_convergence,
)
from porousdiff.virtual_compression import generate_tiled_3d

PHASE_CMAP = ListedColormap(["#0b0e14", "#38bdf8", "#facc15"])


@torch.inference_mode()
def generate_batch(
    model: ContinuousLoadConditionedUNet,
    scheduler: object,
    initial_noise: torch.Tensor,
    load_value: float,
    guidance_scale: float,
    use_amp: bool,
    amp_dtype: torch.dtype,
    material_id: int = 0,
    voxel_size_um: float = 1.0,
    generator: torch.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run reverse diffusion from `initial_noise` `[B, C, D, H, W]`."""
    device = initial_noise.device
    bsz = initial_noise.shape[0]
    image = initial_noise.clone()
    loads = torch.full((bsz,), float(load_value), device=device, dtype=torch.float32)
    mats = (
        torch.full((bsz,), int(material_id), device=device, dtype=torch.long)
        if model.is_multimodal
        else None
    )
    voxs = (
        torch.full((bsz,), float(voxel_size_um), device=device, dtype=torch.float32)
        if model.is_multimodal
        else None
    )

    for timestep in scheduler.timesteps:  # type: ignore[attr-defined]
        t_int = int(timestep)
        t_batch = torch.full((bsz,), t_int, device=device, dtype=torch.long)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            pred = model.forward_cfg(
                image,
                t_batch,
                loads,
                guidance_scale=guidance_scale,
                material_ids=mats,
                voxel_sizes=voxs,
            )
        next_img = scheduler.step(pred.float(), t_int, image, generator=generator)[0]  # type: ignore[attr-defined]
        image = torch.nan_to_num(next_img, nan=0.0, posinf=1.0, neginf=-1.0)

    raw_np = image.float().cpu().numpy()
    phases = decode_channels_to_phases(raw_np, enforce_solid_support=True, threshold=0.0)
    probs = np.clip((raw_np[:, 0] + 1.0) * 0.5, 0.0, 1.0).astype(np.float32)
    return phases, probs


def save_orthoview_figure(
    volume: np.ndarray,
    title: str,
    png_path: Path,
    pdf_path: Path | None = None,
    num_phases: int = 2,
) -> None:
    mid_z, mid_y, mid_x = (s // 2 for s in volume.shape)
    views = [
        (volume[mid_z, :, :], "XY (In-Plane Sheet)"),
        (volume[:, mid_y, :], "XZ (Through-Plane Compression)"),
        (volume[:, :, mid_x], "YZ (Through-Plane Compression)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    cmap = PHASE_CMAP if num_phases == 3 else "gray"
    vmax = 2 if num_phases == 3 else 1
    for ax, (view, vtitle) in zip(axes, views):
        ax.imshow(view, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_title(vtitle)
        ax.axis("off")
    sf = float((volume >= 1).mean())
    extra = f", coat = {float((volume == 2).mean()):.3f}" if num_phases == 3 else ""
    fig.suptitle(f"{title} (solid = {sf:.3f}{extra})")
    fig.tight_layout()
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    if pdf_path is not None:
        fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def save_sweep_panel(
    sweep_volumes: list[tuple[float, np.ndarray]],
    png_path: Path,
    pdf_path: Path,
    num_phases: int = 2,
    unit_label: str = "mV/V",
) -> None:
    n_cols = len(sweep_volumes)
    fig, axes = plt.subplots(2, n_cols, figsize=(3.2 * n_cols, 6.2))
    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    cmap = PHASE_CMAP if num_phases == 3 else "gray"
    vmax = 2 if num_phases == 3 else 1
    for col, (load_val, vol) in enumerate(sweep_volumes):
        mz, my, _ = (s // 2 for s in vol.shape)
        axes[0, col].imshow(vol[mz], cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
        sf = float((vol >= 1).mean())
        axes[0, col].set_title(f"{load_val:g} {unit_label}\nφ_s = {sf:.3f} (XY)")
        axes[0, col].axis("off")

        axes[1, col].imshow(vol[:, my, :], cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
        axes[1, col].set_title("XZ (Compression ↓)")
        axes[1, col].axis("off")

    fig.suptitle("Load-Conditioned Trajectory Sweep (Fixed Initial Gaussian Noise Seed)", fontsize=13)
    fig.tight_layout()
    fig.savefig(png_path, dpi=260, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def _build_eval_specs(
    args: argparse.Namespace,
    cfg: dict[str, object],
) -> list[dict[str, object]]:
    ref_vox = float(cfg.get("ref_voxel_size_um", 1.0))  # type: ignore[arg-type]
    use_kpa = bool(cfg.get("use_stress_kpa", False))
    default_mat = args.material

    if args.manifest_json is not None:
        manifest = json.loads(args.manifest_json.read_text())
        all_scans = list(manifest.get("scans", []))
        filter_set = resolve_manifest_state_subset(all_scans, args.eval_states, "--eval-states")
        specs: list[dict[str, object]] = []
        for scan in all_scans:
            st = str(scan["state"])
            if filter_set is not None and st not in filter_set:
                continue
            if args.beamline and str(scan.get("beamline", "primary")) != args.beamline:
                continue
            if (
                filter_set is None
                and not args.include_repeats
                and bool(scan.get("is_repeat_or_secondary", False))
            ):
                continue
            lv = resolve_scan_load_value(scan, use_kpa)
            specs.append(
                {
                    "state": st,
                    "load_value": lv,
                    "material": str(scan.get("material", "virgin_carbon")),
                    "voxel_size_um": float(scan.get("effective_voxel_size_um", ref_vox)),
                }
            )
        return specs

    if args.eval_states:
        specs = []
        for entry in args.eval_states:
            if "=" in entry:
                k, v = entry.split("=", 1)
                st = k.strip()
                lv = float(v.strip())
            else:
                st = entry.strip()
                lv = parse_load_value(st)
            mat = infer_material_name(st) if default_mat == "auto" else default_mat
            specs.append(
                {
                    "state": st,
                    "load_value": lv,
                    "material": mat,
                    "voxel_size_um": args.voxel_size_um if args.voxel_size_um > 0 else ref_vox,
                }
            )
        return specs

    saved_specs = cfg.get("scan_specs")
    if isinstance(saved_specs, list) and saved_specs:
        return [dict(s) for s in saved_specs]  # type: ignore[arg-type]

    states_map = cfg["states"]  # type: ignore[index]
    return [
        {
            "state": str(k),
            "load_value": float(v),
            "material": infer_material_name(str(k)) if default_mat == "auto" else default_mat,
            "voxel_size_um": args.voxel_size_um if args.voxel_size_um > 0 else ref_vox,
        }
        for k, v in states_map.items()  # type: ignore[union-attr]
    ]


def main() -> None:
    proj_root = default_project_root()
    parser = argparse.ArgumentParser(
        description="Sample load-conditioned 3D microstructures across discrete and interpolated loads"
    )
    parser.add_argument("--experiment", default="carbon_conditional_96_v2")
    parser.add_argument("--output-root", type=Path, default=proj_root)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--manifest-json", type=Path, default=None)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--sampler", choices=["ddim", "ddpm"], default="ddim")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--guidance-scale", type=float, default=1.5)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-states", nargs="+", default=None)
    parser.add_argument("--include-repeats", action="store_true")
    parser.add_argument("--beamline", default=None, help="Optional beamline filter (e.g. DLS_K11_DIAD)")
    parser.add_argument(
        "--material",
        default="auto",
        choices=["auto", "virgin_carbon", "tio2_anode", "cuhcf_cathode"],
    )
    parser.add_argument("--voxel-size-um", type=float, default=0.0, help="0 = use checkpoint/manifest voxel size")
    parser.add_argument(
        "--sweep-loads",
        default="0.008,0.015,0.023,0.033,0.043,0.053",
        help="Comma-separated load/stress values for a fixed-noise sweep (empty to skip)",
    )
    parser.add_argument("--export-openimpala", action="store_true")
    parser.add_argument(
        "--tiled-shape",
        default="",
        help="Optional seamless >patch_size^3 3D MultiDiffusion shape 'D,H,W' (e.g. '96,96,192' or '160,160,160') for OpenImpala REV convergence",
    )
    parser.add_argument("--tiled-overlap", type=float, default=0.5)
    parser.add_argument("--tile-batch-size", type=int, default=4)
    parser.add_argument(
        "--run-rev-study",
        action="store_true",
        help="Run OpenImpala v4.6 REV sub-volume convergence & Z-uniformity study on sample 1 of each state",
    )
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

    ckpt_path = resolve_checkpoint_path(args.output_root, args.experiment, args.checkpoint)
    if ckpt_path is None:
        raise FileNotFoundError(
            f"Checkpoint not found for experiment={args.experiment!r} (requested={args.checkpoint})"
        )

    saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = saved["config"]
    patch_size = int(cfg["patch_size"])
    num_phases = int(cfg.get("num_phases", 2))
    num_materials = int(cfg.get("num_materials", 1))
    ref_vox = float(cfg.get("ref_voxel_size_um", 1.0))
    unit_label = "kPa" if bool(cfg.get("use_stress_kpa", False)) else "mV/V"

    tiled_shape: tuple[int, int, int] | None = None
    if args.tiled_shape.strip():
        parts = [int(p.strip()) for p in args.tiled_shape.split(",") if p.strip()]
        if len(parts) != 3 or any(p < patch_size for p in parts):
            raise ValueError(
                f"--tiled-shape must be 'D,H,W' with each dimension >= patch_size ({patch_size})"
            )
        tiled_shape = (parts[0], parts[1], parts[2])

    model = ContinuousLoadConditionedUNet(
        channels=cfg["channels"],
        attention_levels=cfg["attention_levels"],
        num_res_blocks=int(cfg["num_res_blocks"]),
        norm_num_groups=int(cfg["norm_groups"]),
        load_min=float(cfg["load_min"]),
        load_max=float(cfg["load_max"]),
        use_flash_attention=(device.type == "cuda"),
        conditioning_mode=str(cfg.get("conditioning_mode", "multiscale_fourier")),
        num_phases=num_phases,
        num_materials=num_materials,
        ref_voxel_size_um=ref_vox,
        use_ordinal_material_channel=infer_use_ordinal_material_channel(saved),
        cfg_mode=str(cfg.get("cfg_mode", "all")),
    ).to(device)

    if not args.no_ema and "ema" in saved:
        ema = EMAModel(model)
        ema.load_state_dict(saved["ema"])
        active_model = ema.shadow_model.to(device).eval()
        weight_source = "EMA"
    else:
        model.load_state_dict(saved["model"])
        active_model = model.eval()
        weight_source = "raw"

    scheduler = build_noise_scheduler(
        sampler=args.sampler,
        num_train_timesteps=int(cfg["timesteps"]),
        schedule=str(cfg["schedule"]),
        prediction_type=str(cfg["prediction_type"]),
    )
    scheduler.set_timesteps(args.steps, device=device)

    print(
        f"Loaded {weight_source} weights from {ckpt_path} (epoch {int(saved['epoch']) + 1}) | "
        f"phases={num_phases} | materials={num_materials} | sampler={args.sampler} ({args.steps} steps)"
        + (f" | tiled={tiled_shape}" if tiled_shape else ""),
        flush=True,
    )

    eval_specs = _build_eval_specs(args, cfg)
    samples_subdir = (
        f"samples_tiled_{tiled_shape[0]}x{tiled_shape[1]}x{tiled_shape[2]}"
        if tiled_shape is not None
        else "samples"
    )
    out_dir = args.output_root / "results" / args.experiment / samples_subdir
    figures_dir = args.output_root / "results" / args.experiment / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    per_sample_records: list[dict[str, object]] = []
    summary_records: list[dict[str, object]] = []
    data_ch = active_model.data_channels

    for spec in eval_specs:
        state_name = str(spec["state"])
        load_val = float(spec["load_value"])  # type: ignore[arg-type]
        mat_name = str(spec.get("material", "virgin_carbon"))
        mat_id = resolve_material_id(mat_name) if num_materials > 1 else 0
        vox_um = float(spec.get("voxel_size_um", ref_vox))  # type: ignore[arg-type]

        state_dir = out_dir / state_name
        state_dir.mkdir(parents=True, exist_ok=True)

        gen = torch.Generator(device=device).manual_seed(args.seed)
        collected_vols: list[np.ndarray] = []
        started = time.perf_counter()

        remaining = args.num_samples
        sample_idx = 0
        while remaining > 0:
            cur_b = 1 if tiled_shape is not None else min(args.batch_size, remaining)
            if tiled_shape is not None:
                vols, probs = generate_tiled_3d(
                    model=active_model,
                    scheduler=scheduler,
                    target_shape=tiled_shape,
                    patch_size=patch_size,
                    load_value=load_val,
                    guidance_scale=args.guidance_scale,
                    overlap=args.tiled_overlap,
                    batch_size=cur_b,
                    tile_batch_size=args.tile_batch_size,
                    material_id=mat_id,
                    voxel_size_um=vox_um,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                    generator=gen,
                )
            else:
                noise = torch.randn(
                    (cur_b, data_ch, patch_size, patch_size, patch_size),
                    generator=gen,
                    device=device,
                )
                vols, probs = generate_batch(
                    model=active_model,
                    scheduler=scheduler,
                    initial_noise=noise,
                    load_value=load_val,
                    guidance_scale=args.guidance_scale,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                    material_id=mat_id,
                    voxel_size_um=vox_um,
                    generator=gen,
                )
            for b_vol, p_vol in zip(vols, probs):
                sample_idx += 1
                collected_vols.append(b_vol)
                np.save(state_dir / f"sample_{sample_idx:03d}_binary.npy", b_vol)
                np.save(state_dir / f"sample_{sample_idx:03d}_prob.npy", p_vol)

                if args.export_openimpala:
                    export_openimpala_tiff(
                        b_vol,
                        state_dir / "openimpala" / f"sample_{sample_idx:03d}.tif",
                        preserve_multiphase=(num_phases == 3),
                    )

                phys = (
                    evaluate_multiphase_physics(b_vol, voxel_size=vox_um)
                    if num_phases == 3
                    else evaluate_volume_physics(b_vol, voxel_size=vox_um)
                )
                rec: dict[str, object] = {
                    "state": state_name,
                    "material": mat_name,
                    "load_value": load_val,
                    "voxel_size_um": vox_um,
                    "sample_id": sample_idx,
                }
                rec.update(
                    {k: round(float(v), 6) if np.isfinite(v) else float("nan") for k, v in phys.items()}
                )
                per_sample_records.append(rec)

                if sample_idx == 1:
                    save_orthoview_figure(
                        b_vol,
                        title=f"{state_name} ({mat_name}, {load_val:g} {unit_label})",
                        png_path=figures_dir / f"figure_generated_{state_name}.png",
                        pdf_path=figures_dir / f"figure_generated_{state_name}.pdf",
                        num_phases=num_phases,
                    )
                    if args.run_rev_study:
                        if not has_openimpala_python():
                            raise RuntimeError(
                                "--run-rev-study requires the 'openimpala' Python package."
                            )
                        run_openimpala_rev_convergence(
                            b_vol,
                            phase=0,
                            voxel_size_um=vox_um,
                            out_dir=state_dir / "rev_study",
                            seed=args.seed,
                        )
            remaining -= cur_b

        elapsed = time.perf_counter() - started
        stacked_state = np.stack(collected_vols, axis=0)
        np.save(state_dir / "generated_samples.npy", stacked_state)

        sfs = [float((v >= 1).mean()) for v in collected_vols]
        cfs = [float((v == 2).mean()) for v in collected_vols]
        summary_row = {
            "state": state_name,
            "material": mat_name,
            "load_value": load_val,
            "num_samples": args.num_samples,
            "solid_fraction_mean": round(float(np.mean(sfs)), 6),
            "solid_fraction_std": round(float(np.std(sfs)), 6),
            "coating_fraction_mean": round(float(np.mean(cfs)), 6),
            "sampling_seconds": round(elapsed, 2),
        }
        summary_records.append(summary_row)
        print(json.dumps(summary_row), flush=True)

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

    provenance = {
        "experiment": args.experiment,
        "checkpoint": str(ckpt_path),
        "epoch": int(saved["epoch"]) + 1,
        "weight_source": weight_source,
        "is_pre_registered_ema": weight_source == "ema",
        "sampler": args.sampler,
        "steps": int(args.steps),
        "guidance_scale": float(args.guidance_scale),
        "seed": int(args.seed),
    }
    (out_dir / "sampling_provenance.json").write_text(json.dumps(provenance, indent=2))

    if args.sweep_loads.strip():
        sweep_vals = [float(x.strip()) for x in args.sweep_loads.split(",") if x.strip()]
        if args.material == "auto":
            unique_mats = list(dict.fromkeys(str(s["material"]) for s in eval_specs))
            sweep_mat_name = unique_mats[0] if len(unique_mats) == 1 else "virgin_carbon"
        else:
            sweep_mat_name = args.material
        sweep_mat_id = resolve_material_id(sweep_mat_name) if num_materials > 1 else 0
        sweep_vox = args.voxel_size_um if args.voxel_size_um > 0 else ref_vox

        sweep_gen = torch.Generator(device=device).manual_seed(args.seed + 777)
        shared_noise = torch.randn(
            (1, data_ch, patch_size, patch_size, patch_size),
            generator=sweep_gen,
            device=device,
        )
        sweep_dir = out_dir / "continuous_sweep"
        sweep_dir.mkdir(parents=True, exist_ok=True)
        unit_file_tag = "kPa" if unit_label == "kPa" else "mVV"

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
                material_id=sweep_mat_id,
                voxel_size_um=sweep_vox,
                generator=step_gen,
            )
            vol = b_vol[0]
            tag = f"{mv:.4f}".replace(".", "p")
            np.save(sweep_dir / f"sweep_{tag}{unit_file_tag}.npy", vol)
            sweep_pairs.append((mv, vol))

        save_sweep_panel(
            sweep_pairs,
            png_path=figures_dir / "compression_sweep_orthoviews.png",
            pdf_path=figures_dir / "compression_sweep_orthoviews.pdf",
            num_phases=num_phases,
            unit_label=unit_label,
        )
        print(f"Saved continuous compression sweep ({len(sweep_pairs)} loads) -> {figures_dir}", flush=True)


if __name__ == "__main__":
    main()
