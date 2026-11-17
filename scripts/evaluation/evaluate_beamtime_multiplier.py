"""Head-to-head benchmark: Real Held-Out XCT vs. Numerical Z-Compression Baselines vs. PorousDiff.

Evaluates the core scientific claims of the Beamtime Multiplier paper:
  1. Non-affine compression fidelity: Compares PorousDiff against standard literature heuristics
     (`affine_z_compress_to_target` and `pillar_pore_deletion_compress` computed from each
     material's uncompressed/lowest-load XCT scan) on held-out real synchrotron loads.
  2. Multi-phase coated-electrode electrochemistry: Evaluates 3-phase EASA (`ssa_pore_coating`),
     triple-phase boundary density (`tpb_density`), wired coating fraction, chord anisotropy,
     two-point correlation `S_2(r)` RMSE, percolation, and geodesic pore tortuosity.
  3. Synchrotron Beamtime Multiplier: Compares single-material sparse models, Few-Shot
     Virgin-transferred models (`2` or `3` coated loads), and Full Oracle models side by side.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from porousdiff.baselines import affine_z_compress_to_target, pillar_pore_deletion_compress
from porousdiff.data import (
    default_project_root,
    load_split_patches,
    resolve_material_id,
    resolve_patch_split_dir,
)
from porousdiff.losses import compute_coating_retention_diagnostics
from porousdiff.metrics import (
    directional_two_point_correlation,
    evaluate_multiphase_physics,
    export_openimpala_tiff,
    summarize_metric_records,
)
from porousdiff.models import ContinuousLoadConditionedUNet, EMAModel, build_noise_scheduler
from porousdiff.preprocessing import infer_material_name, parse_load_value
from porousdiff.transport_and_xrd import (
    correlate_xrd_with_microstructure,
    generate_openimpala_job_suite,
    parse_openimpala_results,
)
from porousdiff.virtual_compression import virtual_compress_volumes


def _load_generated_volumes(samples_dir: Path, state_name: str) -> np.ndarray:
    state_dir = samples_dir / state_name
    stacked = state_dir / "generated_samples.npy"
    if stacked.is_file():
        return np.load(stacked)
    files = sorted(state_dir.glob("sample_*_binary.npy"))
    if not files:
        raise FileNotFoundError(f"No generated samples found in {state_dir}")
    return np.stack([np.load(p) for p in files], axis=0)


def _mean_s2_curves(volumes: np.ndarray, max_lag: int) -> dict[str, np.ndarray]:
    per_vol = [directional_two_point_correlation(v >= 1, max_lag=max_lag) for v in volumes]
    return {
        "lags": per_vol[0]["lags"],
        "s2_z": np.mean([c["s2_z"] for c in per_vol], axis=0),
        "s2_xy": np.mean([0.5 * (c["s2_x"] + c["s2_y"]) for c in per_vol], axis=0),
    }


def _s2_rmse(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> float:
    dz = float(np.sqrt(np.mean((a["s2_z"] - b["s2_z"]) ** 2)))
    dxy = float(np.sqrt(np.mean((a["s2_xy"] - b["s2_xy"]) ** 2)))
    return 0.5 * (dz + dxy)


def _parse_experiments(entries: list[str]) -> dict[str, str]:
    """Parse `label=experiment_name` or bare `experiment_name`."""
    out: dict[str, str] = {}
    for item in entries:
        if "=" in item:
            lbl, exp = item.split("=", 1)
            out[lbl.strip()] = exp.strip()
        else:
            out[item.strip()] = item.strip()
    return out


def main() -> None:
    proj_root = default_project_root()
    parser = argparse.ArgumentParser(
        description="Benchmark PorousDiff vs. Affine & Pillar Z-Compression Baselines on Real Held-Out XCT"
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        required=True,
        help="Experiment(s) to evaluate, e.g. porousdiff=beamtime_fewshot oracle=beamtime_oracle",
    )
    parser.add_argument("--data-root", type=Path, default=proj_root / "data" / "carbon_virgin")
    parser.add_argument("--output-root", type=Path, default=proj_root)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--manifest-json", type=Path, default=None)
    parser.add_argument("--version", default="carbon_calibrated_v1")
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--states", nargs="*", default=None, help="Fallback state=load list if no --manifest-json")
    parser.add_argument("--heldout-states", nargs="*", default=None)
    parser.add_argument("--include-repeats", action="store_true")
    parser.add_argument("--beamline", default=None, help="Optional beamline filter (e.g. DLS_K11_DIAD)")
    parser.add_argument("--max-real-samples", type=int, default=20)
    parser.add_argument("--max-openimpala-per-method", type=int, default=5)
    parser.add_argument("--voxel-size-um", type=float, default=1.0)
    parser.add_argument("--export-real-openimpala", action="store_true")
    parser.add_argument(
        "--generate-openimpala-suite",
        action="store_true",
        help="Generate OpenImpala INI configs and run_openimpala_batch.sh across Real, Baselines & PorousDiff",
    )
    parser.add_argument(
        "--xrd-csv",
        type=Path,
        default=None,
        help="Optional DLS K11 DIAD co-registered XRD CSV to correlate against 3D microstructural observables",
    )
    parser.add_argument(
        "--include-virtual-compression",
        action="store_true",
        help="Virtually compress real lowest-load XCT patches via deterministic DDIM ODE inversion + target-stress denoising",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional checkpoint override for --include-virtual-compression")
    parser.add_argument("--virtual-steps", type=int, default=40)
    parser.add_argument("--invert-strength", type=float, default=0.85)
    parser.add_argument("--guidance-scale", type=float, default=1.5)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    exp_map = _parse_experiments(args.experiments)
    first_exp = next(iter(exp_map.values()))
    out_dir = (
        args.out_dir
        if args.out_dir is not None
        else (args.output_root / "results" / first_exp / "beamtime_benchmark")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    heldout_set = set(args.heldout_states or [])

    # Build state specification list grouped by (material, beamline)
    state_specs: list[dict[str, object]] = []
    if args.manifest_json is not None:
        manifest = json.loads(args.manifest_json.read_text())
        for sc in manifest.get("scans", []):
            if not args.include_repeats and bool(sc.get("is_repeat_or_secondary", False)):
                continue
            sc_beam = str(sc.get("beamline", "primary"))
            if args.beamline and sc_beam != args.beamline:
                continue
            state_specs.append(
                {
                    "state": str(sc["state"]),
                    "material": str(sc.get("material", "virgin_carbon")),
                    "beamline": sc_beam,
                    "load_value": float(sc.get("stress_kpa") or sc["load_mv_per_v"]),
                    "voxel_size_um": float(sc.get("effective_voxel_size_um", args.voxel_size_um)),
                    "num_phases": int(sc.get("num_phases", 2)),
                    "patch_dir_name": sc.get("patch_dir_name"),
                }
            )
    elif args.states:
        for item in args.states:
            if "=" in item:
                st, lv_s = item.split("=", 1)
                st, lv = st.strip(), float(lv_s.strip())
            else:
                st = item.strip()
                lv = parse_load_value(st)
            state_specs.append(
                {
                    "state": st,
                    "material": infer_material_name(st),
                    "beamline": args.beamline or "primary",
                    "load_value": lv,
                    "voxel_size_um": args.voxel_size_um,
                    "num_phases": 3 if infer_material_name(st) != "virgin_carbon" else 2,
                    "patch_dir_name": None,
                }
            )
    else:
        raise ValueError("Provide either --manifest-json or --states")

    state_specs.sort(
        key=lambda s: (str(s["material"]), str(s["beamline"]), float(s["load_value"]))  # type: ignore[arg-type]
    )

    # Identify lowest-load reference real validation patches per (material, beamline)
    rng = np.random.default_rng(args.seed)
    expected_shape = (args.patch_size, args.patch_size, args.patch_size)
    real_cache: dict[str, np.ndarray] = {}
    lowest_state_by_group: dict[tuple[str, str], str] = {}
    lowest_load_by_group: dict[tuple[str, str], float] = {}

    for spec in state_specs:
        st = str(spec["state"])
        mat = str(spec["material"])
        beam = str(spec["beamline"])
        lv = float(spec["load_value"])  # type: ignore[arg-type]
        pdir_hint = str(spec["patch_dir_name"]) if spec.get("patch_dir_name") else None
        split_dir = resolve_patch_split_dir(
            args.data_root, st, args.patch_size, args.version, "val", patch_dir_name=pdir_hint
        )
        vols = load_split_patches(split_dir, expected_shape=expected_shape)
        if vols.shape[0] > args.max_real_samples:
            idx = np.sort(rng.choice(vols.shape[0], size=args.max_real_samples, replace=False))
            vols = vols[idx]
        real_cache[st] = vols
        if (mat, beam) not in lowest_state_by_group:
            lowest_state_by_group[(mat, beam)] = st
            lowest_load_by_group[(mat, beam)] = lv

    # Optional: Virtual in-situ compression of real uncompressed scans via DDIM ODE inversion
    virtual_cache_by_state: dict[str, np.ndarray] = {}
    if args.include_virtual_compression:
        if torch.cuda.is_available():
            vc_device = torch.device("cuda")
        elif args.allow_cpu:
            vc_device = torch.device("cpu")
        else:
            raise RuntimeError("CUDA GPU required for --include-virtual-compression (or pass --allow-cpu)")

        ckpt_dir = args.output_root / "checkpoints" / first_exp
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
            raise FileNotFoundError(f"Checkpoint not found for virtual compression: {ckpt_path}")

        saved_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        vc_cfg = saved_ckpt["config"]
        vc_num_materials = int(vc_cfg.get("num_materials", 1))
        vc_model = ContinuousLoadConditionedUNet(
            channels=vc_cfg["channels"],
            attention_levels=vc_cfg["attention_levels"],
            num_res_blocks=int(vc_cfg["num_res_blocks"]),
            norm_num_groups=int(vc_cfg["norm_groups"]),
            load_min=float(vc_cfg["load_min"]),
            load_max=float(vc_cfg["load_max"]),
            use_flash_attention=(vc_device.type == "cuda"),
            conditioning_mode=str(vc_cfg.get("conditioning_mode", "multiscale_fourier")),
            num_phases=int(vc_cfg.get("num_phases", 2)),
            num_materials=vc_num_materials,
            ref_voxel_size_um=float(vc_cfg.get("ref_voxel_size_um", 1.0)),
        ).to(vc_device)
        if "ema" in saved_ckpt:
            vc_ema = EMAModel(vc_model)
            vc_ema.load_state_dict(saved_ckpt["ema"])
            vc_active = vc_ema.shadow_model.to(vc_device).eval()
        else:
            vc_model.load_state_dict(saved_ckpt["model"])
            vc_active = vc_model.eval()

        vc_sched = build_noise_scheduler(
            sampler="ddim",
            num_train_timesteps=int(vc_cfg["timesteps"]),
            schedule=str(vc_cfg["schedule"]),
            prediction_type=str(vc_cfg["prediction_type"]),
        )
        vc_use_amp = vc_device.type == "cuda"
        vc_amp_dtype = (
            torch.bfloat16
            if (vc_device.type == "cuda" and torch.cuda.is_bf16_supported())
            else torch.float16
        )

        specs_by_group: dict[tuple[str, str], list[dict[str, object]]] = {}
        for sp in state_specs:
            specs_by_group.setdefault((str(sp["material"]), str(sp["beamline"])), []).append(sp)

        for (mat, beam), grp_specs in specs_by_group.items():
            ref_st = lowest_state_by_group[(mat, beam)]
            ref_lv = lowest_load_by_group[(mat, beam)]
            ref_vols = real_cache[ref_st]
            target_lvs = [float(sp["load_value"]) for sp in grp_specs]  # type: ignore[arg-type]
            mat_id = resolve_material_id(mat) if vc_num_materials > 1 else 0
            vox_um = float(grp_specs[0]["voxel_size_um"])  # type: ignore[arg-type]
            vc_map = virtual_compress_volumes(
                model=vc_active,
                alphas_cumprod=vc_sched.alphas_cumprod,
                source_volumes=ref_vols,
                source_load=ref_lv,
                target_loads=target_lvs,
                prediction_type=str(vc_cfg["prediction_type"]),
                num_steps=args.virtual_steps,
                invert_strength=args.invert_strength,
                guidance_scale=args.guidance_scale,
                material_id=mat_id,
                voxel_size_um=vox_um,
                batch_size=min(4, int(ref_vols.shape[0])),
                use_amp=vc_use_amp,
                amp_dtype=vc_amp_dtype,
            )
            for sp in grp_specs:
                st_k = str(sp["state"])
                lv_k = float(sp["load_value"])  # type: ignore[arg-type]
                virtual_cache_by_state[st_k] = vc_map[lv_k]

    max_lag = min(32, args.patch_size // 2)
    report: dict[str, object] = {}
    csv_rows: list[dict[str, object]] = []
    openimpala_entries: list[dict[str, object]] = []
    want_openimpala = args.export_real_openimpala or args.generate_openimpala_suite

    key_metrics = [
        "solid_fraction",
        "coating_fraction",
        "ssa_pore_coating",
        "tpb_density",
        "wired_coating_fraction",
        "coating_pinch_fraction",
        "pore_anisotropy_xy_over_z",
        "geodesic_tortuosity_z",
    ]

    for spec in state_specs:
        st = str(spec["state"])
        mat = str(spec["material"])
        beam = str(spec["beamline"])
        lv = float(spec["load_value"])  # type: ignore[arg-type]
        vox = float(spec["voxel_size_um"])  # type: ignore[arg-type]
        n_ph = int(spec["num_phases"])  # type: ignore[arg-type]
        is_heldout = st in heldout_set

        real_vols = real_cache[st]
        if want_openimpala:
            for r_i, rv in enumerate(real_vols[: args.max_openimpala_per_method], start=1):
                tif_p = out_dir / "openimpala_real" / st / f"real_val_{r_i:03d}.tif"
                export_openimpala_tiff(rv, tif_p, preserve_multiphase=(n_ph == 3))
                openimpala_entries.append(
                    {
                        "method": "real",
                        "state": st,
                        "material": mat,
                        "beamline": beam,
                        "load_value": lv,
                        "tiff_path": tif_p,
                        "num_phases": n_ph,
                    }
                )

        real_recs = [evaluate_multiphase_physics(v, voxel_size=vox) for v in real_vols]
        real_sum = summarize_metric_records(real_recs)
        real_s2 = _mean_s2_curves(real_vols, max_lag=max_lag)
        target_sf = float(real_sum["solid_fraction"]["mean"])

        # Construct literature Z-compression baselines from uncompressed scan of same (material, beamline)
        ref_st = lowest_state_by_group[(mat, beam)]
        ref_vols = real_cache[ref_st]
        affine_vols = np.stack(
            [
                affine_z_compress_to_target(
                    rv, target_solid_fraction=max(target_sf, 1e-3), output_depth=args.patch_size
                )
                for rv in ref_vols
            ],
            axis=0,
        )
        pillar_vols = np.stack(
            [
                pillar_pore_deletion_compress(
                    rv,
                    target_solid_fraction=max(target_sf, 1e-3),
                    output_depth=args.patch_size,
                    rng=np.random.default_rng(args.seed + i),
                )
                for i, rv in enumerate(ref_vols)
            ],
            axis=0,
        )

        methods_vols: dict[str, np.ndarray] = {
            "affine_z_squash": affine_vols,
            "pillar_pore_deletion": pillar_vols,
        }
        if st in virtual_cache_by_state:
            methods_vols["virtual_compression"] = virtual_cache_by_state[st]
        for lbl, exp_name in exp_map.items():
            samples_dir = args.output_root / "results" / exp_name / "samples"
            if (samples_dir / st).is_dir():
                methods_vols[lbl] = _load_generated_volumes(samples_dir, st)

        if want_openimpala:
            for m_name, m_vols in methods_vols.items():
                for b_i, mv_vol in enumerate(m_vols[: args.max_openimpala_per_method], start=1):
                    tif_p = out_dir / "openimpala_methods" / m_name / st / f"{m_name}_{b_i:03d}.tif"
                    export_openimpala_tiff(mv_vol, tif_p, preserve_multiphase=(n_ph == 3))
                    openimpala_entries.append(
                        {
                            "method": m_name,
                            "state": st,
                            "material": mat,
                            "beamline": beam,
                            "load_value": lv,
                            "tiff_path": tif_p,
                            "num_phases": n_ph,
                        }
                    )

        methods_summary: dict[str, object] = {}
        for m_name, m_vols in methods_vols.items():
            m_recs = [evaluate_multiphase_physics(v, voxel_size=vox) for v in m_vols]
            m_sum = summarize_metric_records(m_recs)
            m_s2 = _mean_s2_curves(m_vols, max_lag=max_lag)
            s2_err = _s2_rmse(real_s2, m_s2)

            methods_summary[m_name] = {
                "num_volumes": int(m_vols.shape[0]),
                "s2_rmse": s2_err,
                "metrics": m_sum,
            }

            row: dict[str, object] = {
                "state": st,
                "material": mat,
                "beamline": beam,
                "load_value": lv,
                "is_heldout": is_heldout,
                "method": m_name,
                "s2_rmse": round(s2_err, 6),
            }
            for km in key_metrics:
                r_mu = real_sum[km]["mean"]
                m_mu = m_sum[km]["mean"]
                row[f"real_{km}"] = round(r_mu, 6) if np.isfinite(r_mu) else float("nan")
                row[f"pred_{km}"] = round(m_mu, 6) if np.isfinite(m_mu) else float("nan")
                row[f"abs_err_{km}"] = (
                    round(abs(m_mu - r_mu), 6)
                    if (np.isfinite(r_mu) and np.isfinite(m_mu))
                    else float("nan")
                )
            csv_rows.append(row)

        report[st] = {
            "material": mat,
            "beamline": beam,
            "load_value": lv,
            "voxel_size_um": vox,
            "is_heldout": is_heldout,
            "reference_uncompressed_state": ref_st,
            "real": {"num_volumes": int(real_vols.shape[0]), "metrics": real_sum},
            "methods": methods_summary,
        }
        print(
            f"[{st} | {mat} ({beam}) | heldout={is_heldout}] real φ_s={target_sf:.4f} | "
            + " | ".join(
                f"{m}: s2_rmse={methods_summary[m]['s2_rmse']:.4f}"  # type: ignore[index]
                for m in methods_summary
            ),
            flush=True,
        )

    retention_diag = compute_coating_retention_diagnostics(report)
    with (out_dir / "coating_retention_diagnostics.json").open("w") as f:
        json.dump(retention_diag, f, indent=2)

    with (out_dir / "beamtime_multiplier_report.json").open("w") as f:
        json.dump(report, f, indent=2)

    if csv_rows:
        with (out_dir / "baseline_vs_porousdiff_table.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    if want_openimpala and openimpala_entries:
        suite_info = generate_openimpala_job_suite(
            tiff_entries=openimpala_entries,
            suite_dir=out_dir / "openimpala_suite",
        )
        print(
            f"Generated OpenImpala batch suite ({suite_info['num_jobs']} jobs) -> {suite_info['script_path']}",
            flush=True,
        )
        if (out_dir / "openimpala_suite" / "outputs").is_dir():
            parse_openimpala_results(
                suite_dir=out_dir / "openimpala_suite",
                out_csv=out_dir / "openimpala_transport_parity.csv",
            )

    if args.xrd_csv is not None:
        xrd_summary = correlate_xrd_with_microstructure(
            xrd_csv_path=args.xrd_csv,
            benchmark_report=report,
            out_dir=out_dir / "diad_xrd_analysis",
        )
        print(
            f"Saved DIAD XRD correlation ({xrd_summary['num_matched_states']} states) -> {out_dir / 'diad_xrd_analysis'}",
            flush=True,
        )

    # Plot Real XCT vs Baselines vs PorousDiff across loads per material
    materials = sorted({str(s["material"]) for s in state_specs})
    fig, axes = plt.subplots(len(materials), 3, figsize=(14, 4.0 * len(materials)), squeeze=False)
    plot_Specs = [
        ("pore_anisotropy_xy_over_z", "Pore Chord Anisotropy (XY / Z)"),
        ("geodesic_tortuosity_z", "Through-Plane Geodesic Tortuosity τ_Z"),
        ("ssa_pore_coating", "Active Coating EASA (1/µm)"),
    ]

    for r_idx, mat in enumerate(materials):
        mat_states = [str(s["state"]) for s in state_specs if str(s["material"]) == mat]
        loads = [float(report[st]["load_value"]) for st in mat_states]  # type: ignore[index]

        for c_idx, (m_key, m_label) in enumerate(plot_Specs):
            ax = axes[r_idx, c_idx]
            real_y = [report[st]["real"]["metrics"][m_key]["mean"] for st in mat_states]  # type: ignore[index]
            ax.plot(loads, real_y, "-o", color="black", lw=2.0, label="Real XCT (Val)")

            all_methods = list(report[mat_states[0]]["methods"].keys())  # type: ignore[index]
            palette = ["tab:gray", "tab:orange", "tab:blue", "tab:green", "tab:red"]
            for m_i, m_name in enumerate(all_methods):
                my = [
                    report[st]["methods"][m_name]["metrics"][m_key]["mean"]  # type: ignore[index]
                    for st in mat_states
                    if m_name in report[st]["methods"]  # type: ignore[index]
                ]
                mx = [
                    float(report[st]["load_value"])  # type: ignore[index]
                    for st in mat_states
                    if m_name in report[st]["methods"]  # type: ignore[index]
                ]
                style = "--s" if m_name in ("affine_z_squash", "pillar_pore_deletion") else "-^"
                ax.plot(mx, my, style, color=palette[m_i % len(palette)], lw=1.5, label=m_name)

            ax.set_title(f"{mat} — {m_label}", fontsize=10)
            ax.set_xlabel("Load / Stress")
            ax.grid(True, alpha=0.3)
            if r_idx == 0 and c_idx == 0:
                ax.legend(fontsize=8)

    fig.suptitle("Real Held-Out Synchrotron XCT vs. Numerical Z-Compression vs. PorousDiff", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "beamtime_multiplier_comparison.png", dpi=240, bbox_inches="tight")
    fig.savefig(out_dir / "beamtime_multiplier_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Beamtime Multiplier benchmark -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
