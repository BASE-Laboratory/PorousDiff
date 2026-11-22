"""Virtual in-situ compression of a real uncompressed (0 kPa / benchtop XCT) 3D scan.

Deterministically inverts an actual segmented 3D electrode volume (`{0, 1}` or `{0, 1, 2}`)
at `--source-load` along the conditional DDIM probability-flow ODE and denoises it back to
`t = 0` at any list of `--target-loads`, producing:
  - Virtually compressed 3D `.npy` volumes and optional OpenImpala `.tif` stacks,
  - `virtual_compression_metrics.csv` quantifying porosity, EASA, TPB, pinch-points, and `tau_Z`,
  - Publication side-by-side comparison figure `virtual_compression_panel.{png,pdf}`.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
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

from porousdiff.data import default_project_root, resolve_material_id
from porousdiff.io import export_volume_ome_zarr, load_volume_array
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
from porousdiff.virtual_compression import (
    check_load_extrapolation,
    compute_virtual_compression_fidelity,
    virtual_compress_lagrangian_slab,
    virtual_compress_volumes,
)

PHASE_CMAP = ListedColormap(["#0b0e14", "#38bdf8", "#facc15"])


def _load_input_volume(path: Path) -> tuple[np.ndarray, float | None]:
    return load_volume_array(path)


def _save_virtual_panel(
    source_vol: np.ndarray,
    source_load: float,
    compressed_by_load: dict[float, np.ndarray],
    png_path: Path,
    pdf_path: Path,
    num_phases: int,
    unit_label: str,
) -> None:
    items = [(source_load, source_vol, "Real Input")] + [
        (lv, vol, "Virtual PorousDiff") for lv, vol in sorted(compressed_by_load.items())
    ]
    n_cols = len(items)
    fig, axes = plt.subplots(2, n_cols, figsize=(3.3 * n_cols, 6.2), squeeze=False)
    cmap = PHASE_CMAP if num_phases == 3 else "gray"
    vmax = 2 if num_phases == 3 else 1

    for col, (lv, vol, tag) in enumerate(items):
        mz, my, _ = (s // 2 for s in vol.shape)
        sf = float((vol >= 1).mean())
        axes[0, col].imshow(vol[mz], cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
        axes[0, col].set_title(f"{tag}\n{lv:g} {unit_label} (φ_s={sf:.3f})", fontsize=10)
        axes[0, col].axis("off")

        axes[1, col].imshow(vol[:, my, :], cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
        axes[1, col].set_title(f"XZ (Z={vol.shape[0]} ↓)", fontsize=9)
        axes[1, col].axis("off")

    fig.suptitle("Conditional DDIM Microstructure Transfer Across Compressive Loads", fontsize=12)
    fig.tight_layout()
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    proj_root = default_project_root()
    parser = argparse.ArgumentParser(
        description="Transfer a real low-load 3D XCT scan across target compressive loads via DDIM ODE inversion"
    )
    parser.add_argument("--input", type=Path, required=True, help="Input 3D segmented .npy or .tif scan")
    parser.add_argument("--experiment", default="beamtime_multiplier_96")
    parser.add_argument("--output-root", type=Path, default=proj_root)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--source-load", type=float, required=True, help="Physical load/stress of the input scan")
    parser.add_argument(
        "--target-loads",
        required=True,
        help="Comma-separated target loads/stresses (e.g. '50,150,300,500' or '0.018,0.033,0.053')",
    )
    parser.add_argument(
        "--mode",
        choices=["eulerian_ddim", "lagrangian_slab"],
        default="eulerian_ddim",
        help="eulerian_ddim: fixed-box DDIM ODE inversion; lagrangian_slab: mass-conserving Z-shrinking slab MultiDiffusion",
    )
    parser.add_argument(
        "--target-compression-ratios",
        default="",
        help="Optional comma-separated H_comp/H_0 ratios matching --target-loads when --mode lagrangian_slab",
    )
    parser.add_argument(
        "--material",
        default="virgin_carbon",
        choices=["virgin_carbon", "tio2_anode", "cuhcf_cathode"],
    )
    parser.add_argument("--voxel-size-um", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--invert-strength", type=float, default=0.60)
    parser.add_argument("--refine-strength", type=float, default=0.35)
    parser.add_argument("--guidance-scale", type=float, default=1.5)
    parser.add_argument("--export-openimpala", action="store_true")
    parser.add_argument("--export-ome-zarr", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.allow_cpu:
        device = torch.device("cpu")
    else:
        raise RuntimeError("CUDA GPU required (pass --allow-cpu for local CPU testing).")

    ckpt_dir = args.output_root / "checkpoints" / args.experiment
    ckpt_path = (
        args.checkpoint
        if args.checkpoint is not None
        else (
            ckpt_dir / "best_val.pth"
            if (ckpt_dir / "best_val.pth").is_file()
            else ckpt_dir / "latest.pth"
        )
    )
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    source_vol, loaded_vox_um = _load_input_volume(args.input)

    saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = saved["config"]
    patch_size = int(cfg.get("patch_size", 96))
    num_phases = int(cfg.get("num_phases", 2))
    num_materials = int(cfg.get("num_materials", 1))
    ref_vox = float(cfg.get("ref_voxel_size_um", 1.0))
    vox_um = (
        args.voxel_size_um
        if args.voxel_size_um > 0
        else (loaded_vox_um if loaded_vox_um is not None else ref_vox)
    )
    unit_label = "kPa" if bool(cfg.get("use_stress_kpa", False)) else "mV/V"

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
    if "ema" in saved:
        ema = EMAModel(model)
        ema.load_state_dict(saved["ema"])
        active_model = ema.shadow_model.to(device).eval()
    else:
        model.load_state_dict(saved["model"])
        active_model = model.eval()

    scheduler = build_noise_scheduler(
        sampler="ddim",
        num_train_timesteps=int(cfg["timesteps"]),
        schedule=str(cfg["schedule"]),
        prediction_type=str(cfg["prediction_type"]),
    )

    target_loads = [float(x.strip()) for x in args.target_loads.split(",") if x.strip()]
    extrap = check_load_extrapolation(
        [float(args.source_load), *target_loads],
        float(cfg["load_min"]),
        float(cfg["load_max"]),
    )
    if bool(extrap["is_extrapolated"]):
        print(
            f"Warning: Requested loads {extrap['extrapolated_loads']} extrapolate outside trained range "
            f"[{extrap['load_min']}, {extrap['load_max']}] ({unit_label}).",
            flush=True,
        )

    mat_id = resolve_material_id(args.material) if num_materials > 1 else 0
    use_amp = device.type == "cuda"
    amp_dtype = (
        torch.bfloat16
        if (device.type == "cuda" and torch.cuda.is_bf16_supported())
        else torch.float16
    )

    lagrangian_info_by_load: dict[str, dict[str, float]] = {}
    if args.mode == "lagrangian_slab":
        if args.target_compression_ratios.strip():
            ratios = [float(x.strip()) for x in args.target_compression_ratios.split(",") if x.strip()]
            if len(ratios) != len(target_loads):
                raise ValueError("--target-compression-ratios must match length of --target-loads")
        else:
            l_min = float(cfg["load_min"])
            l_span = max(float(cfg["load_max"]) - l_min, 1e-6)
            ratios = [
                float(np.clip(1.0 - 0.45 * max(0.0, tl - float(args.source_load)) / l_span, 0.40, 1.0))
                for tl in target_loads
            ]
        same_load_vol, _ = virtual_compress_lagrangian_slab(
            model=active_model,
            alphas_cumprod=scheduler.alphas_cumprod,
            source_volume=source_vol,
            target_load=float(args.source_load),
            patch_size=patch_size,
            compression_ratio=1.0,
            prediction_type=str(cfg["prediction_type"]),
            num_steps=args.steps,
            refine_strength=args.refine_strength,
            guidance_scale=1.0,
            material_id=mat_id,
            voxel_size_um=vox_um,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )
        compressed = {}
        for tl, r_val in zip(target_loads, ratios):
            c_vol, l_info = virtual_compress_lagrangian_slab(
                model=active_model,
                alphas_cumprod=scheduler.alphas_cumprod,
                source_volume=source_vol,
                target_load=float(tl),
                patch_size=patch_size,
                compression_ratio=float(r_val),
                prediction_type=str(cfg["prediction_type"]),
                num_steps=args.steps,
                refine_strength=args.refine_strength,
                guidance_scale=args.guidance_scale,
                material_id=mat_id,
                voxel_size_um=vox_um,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )
            compressed[float(tl)] = c_vol
            lagrangian_info_by_load[f"{float(tl):g}"] = l_info
    else:
        all_eval_loads = list(dict.fromkeys([float(args.source_load), *target_loads]))
        all_compressed = virtual_compress_volumes(
            model=active_model,
            alphas_cumprod=scheduler.alphas_cumprod,
            source_volumes=source_vol,
            source_load=args.source_load,
            target_loads=all_eval_loads,
            prediction_type=str(cfg["prediction_type"]),
            num_steps=args.steps,
            invert_strength=args.invert_strength,
            guidance_scale=args.guidance_scale,
            material_id=mat_id,
            voxel_size_um=vox_um,
            batch_size=1,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )
        same_load_vol = all_compressed[float(args.source_load)]
        compressed = {float(tl): all_compressed[float(tl)] for tl in target_loads}

    out_dir = (
        args.out_dir
        if args.out_dir is not None
        else (args.output_root / "results" / args.experiment / "virtual_compression")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    fidelity_by_load: dict[str, dict[str, float | bool]] = {}
    eval_fn = evaluate_multiphase_physics if num_phases == 3 else evaluate_volume_physics
    src_phys = eval_fn(source_vol, voxel_size=vox_um)
    src_fid = compute_virtual_compression_fidelity(
        source_vol, same_load_vol, same_load_reconstructions=same_load_vol
    )
    src_row: dict[str, object] = {
        "kind": "real_source",
        "mode": args.mode,
        "depth_z": int(source_vol.shape[0]),
        "load_value": args.source_load,
        "load_unit": unit_label,
        "material": args.material,
        "same_load_solid_dice": round(float(src_fid["same_load_solid_dice"]), 6),
        "paired_xy_column_corr": 1.0,
    }
    src_row.update({k: round(float(v), 6) if np.isfinite(v) else float("nan") for k, v in src_phys.items()})
    rows.append(src_row)

    for lv, vol in sorted(compressed.items()):
        tag = f"{lv:.4f}".replace(".", "p")
        np.save(out_dir / f"virtual_{args.material}_{tag}.npy", vol)
        if args.export_openimpala:
            export_openimpala_tiff(
                vol,
                out_dir / "openimpala" / f"virtual_{args.material}_{tag}.tif",
                preserve_multiphase=(num_phases == 3),
            )
        if args.export_ome_zarr:
            export_volume_ome_zarr(
                vol,
                out_dir / "ome_zarr" / f"virtual_{args.material}_{tag}.ome.zarr",
                voxel_size_um=vox_um,
                preserve_multiphase=(num_phases == 3),
                metadata={
                    "material": args.material,
                    "load_value": float(lv),
                    "load_unit": unit_label,
                    "mode": args.mode,
                },
            )
        phys = eval_fn(vol, voxel_size=vox_um)
        fid = compute_virtual_compression_fidelity(
            source_vol, vol, same_load_reconstructions=same_load_vol
        )
        fidelity_by_load[f"{lv:g}"] = fid
        row: dict[str, object] = {
            "kind": "virtual_compressed",
            "mode": args.mode,
            "depth_z": int(vol.shape[0]),
            "load_value": lv,
            "load_unit": unit_label,
            "material": args.material,
            "same_load_solid_dice": round(float(fid["same_load_solid_dice"]), 6),
            "paired_xy_column_corr": round(float(fid["paired_xy_column_corr"]), 6),
        }
        row.update({k: round(float(v), 6) if np.isfinite(v) else float("nan") for k, v in phys.items()})
        rows.append(row)

    with (out_dir / "virtual_compression_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_payload = {
        "mode": args.mode,
        "load_unit": unit_label,
        "invert_strength": float(args.invert_strength),
        "refine_strength": float(args.refine_strength),
        "extrapolation_check": extrap,
        "same_load_solid_dice": round(float(src_fid["same_load_solid_dice"]), 6),
        "fidelity_by_load": fidelity_by_load,
        "lagrangian_slab_diagnostics": lagrangian_info_by_load,
        "records": rows,
    }
    (out_dir / "virtual_compression_summary.json").write_text(json.dumps(summary_payload, indent=2))
    _save_virtual_panel(
        source_vol=source_vol,
        source_load=args.source_load,
        compressed_by_load=compressed,
        png_path=out_dir / "virtual_compression_panel.png",
        pdf_path=out_dir / "virtual_compression_panel.pdf",
        num_phases=num_phases,
        unit_label=unit_label,
    )
    print(f"Saved virtual compression series ({len(compressed)} target loads) -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
