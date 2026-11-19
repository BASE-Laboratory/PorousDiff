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

from porousdiff.baselines import (
    affine_z_compress_to_target,
    compute_mirror_tiled_fraction,
    pillar_pore_deletion_compress,
)
from porousdiff.data import (
    default_project_root,
    load_split_patches,
    load_tall_or_split_patches,
    resolve_checkpoint_path,
    resolve_manifest_state_subset,
    resolve_material_id,
    resolve_patch_split_dir,
    resolve_scan_load_value,
    resolve_use_stress_kpa_from_checkpoints,
)
from porousdiff.losses import compute_coating_retention_diagnostics
from porousdiff.metrics import (
    compute_ensemble_distribution_distances,
    directional_two_point_correlation,
    evaluate_multiphase_physics,
    export_openimpala_tiff,
    summarize_metric_records,
)
from porousdiff.models import (
    ContinuousLoadConditionedUNet,
    EMAModel,
    build_noise_scheduler,
    infer_use_ordinal_material_channel,
)
from porousdiff.preprocessing import infer_material_name, parse_load_value
from porousdiff.transport_and_xrd import (
    correlate_xrd_with_microstructure,
    generate_openimpala_job_suite,
    has_openimpala_python,
    parse_openimpala_results,
    run_openimpala_python_suite,
)
from porousdiff.virtual_compression import (
    check_load_extrapolation,
    compute_virtual_compression_fidelity,
    virtual_compress_volumes,
)


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


def _parse_experiments(entries: list[str]) -> dict[str, list[str]]:
    """Parse `label=exp_s1,exp_s2,...` (or `+`-separated) or bare `experiment_name`."""
    out: dict[str, list[str]] = {}
    for item in entries:
        if "=" in item:
            lbl, exp_raw = item.split("=", 1)
            exps = [x.strip() for x in exp_raw.replace("+", ",").split(",") if x.strip()]
            out[lbl.strip()] = exps
        else:
            exps = [x.strip() for x in item.replace("+", ",").split(",") if x.strip()]
            if exps:
                out[exps[0]] = exps
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
        help="Experiment(s) to evaluate; pass comma-separated seeds for error bars (e.g. porousdiff=exp_s42,exp_s43,exp_s44)",
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
    parser.add_argument(
        "--use-stress-kpa",
        action="store_true",
        help="Fallback load unit when no checkpoint exists under --output-root (otherwise read from checkpoint config)",
    )
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
        "--run-openimpala",
        action="store_true",
        help="Execute OpenImpala v4.6 PDE transport directly in-process and write openimpala_transport_parity.csv",
    )
    parser.add_argument(
        "--openimpala-template-ini",
        type=Path,
        default=None,
        help="Optional user OpenImpala INI template with {tiff_path}, {phase}, {transport_phase}, {output_prefix}",
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
    parser.add_argument("--invert-strength", type=float, default=0.60)
    parser.add_argument("--guidance-scale", type=float, default=1.5)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    exp_map = _parse_experiments(args.experiments)
    all_exp_names = [e for exps in exp_map.values() for e in exps]
    first_exp = all_exp_names[0]
    out_dir = (
        args.out_dir
        if args.out_dir is not None
        else (args.output_root / "results" / first_exp / "beamtime_benchmark")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    heldout_set = set(args.heldout_states or [])

    use_kpa, _ = resolve_use_stress_kpa_from_checkpoints(
        output_root=args.output_root,
        experiments=all_exp_names,
        explicit_checkpoint=args.checkpoint,
        default=args.use_stress_kpa,
    )
    unit_label = "kPa" if use_kpa else "mV/V"

    # Build state specification list grouped by (material, beamline)
    state_specs: list[dict[str, object]] = []
    if args.manifest_json is not None:
        manifest = json.loads(args.manifest_json.read_text())
        all_manifest_scans = list(manifest.get("scans", []))
        if all_manifest_scans and args.patch_size == 96 and "patch_size" in all_manifest_scans[0]:
            args.patch_size = int(all_manifest_scans[0]["patch_size"])
        if args.heldout_states:
            resolved_held = resolve_manifest_state_subset(
                all_manifest_scans, args.heldout_states, "--heldout-states"
            )
            assert resolved_held is not None
            heldout_set = resolved_held
        for sc in all_manifest_scans:
            if not args.include_repeats and bool(sc.get("is_repeat_or_secondary", False)):
                continue
            sc_beam = str(sc.get("beamline", "primary"))
            if args.beamline and sc_beam != args.beamline:
                continue
            lv_resolved = resolve_scan_load_value(sc, use_kpa)
            state_specs.append(
                {
                    "state": str(sc["state"]),
                    "material": str(sc.get("material", "virgin_carbon")),
                    "beamline": sc_beam,
                    "load_value": lv_resolved,
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
    tall_ref_cache: dict[str, np.ndarray] = {}
    lowest_state_by_group: dict[tuple[str, str], str] = {}
    lowest_load_by_group: dict[tuple[str, str], float] = {}
    manifest_fallback_root = args.manifest_json.parent if args.manifest_json is not None else None

    for spec in state_specs:
        st = str(spec["state"])
        mat = str(spec["material"])
        beam = str(spec["beamline"])
        lv = float(spec["load_value"])  # type: ignore[arg-type]
        pdir_hint = str(spec["patch_dir_name"]) if spec.get("patch_dir_name") else None
        split_dir = resolve_patch_split_dir(
            args.data_root,
            st,
            args.patch_size,
            args.version,
            "val",
            patch_dir_name=pdir_hint,
            fallback_root=manifest_fallback_root,
        )
        vols = load_split_patches(split_dir, expected_shape=expected_shape)
        if vols.shape[0] > args.max_real_samples:
            idx = np.sort(rng.choice(vols.shape[0], size=args.max_real_samples, replace=False))
            vols = vols[idx]
        real_cache[st] = vols
        tall_vols = load_tall_or_split_patches(split_dir, vols)
        if tall_vols.shape[0] > args.max_real_samples:
            tall_vols = tall_vols[: args.max_real_samples]
        tall_ref_cache[st] = tall_vols
        if (mat, beam) not in lowest_state_by_group:
            lowest_state_by_group[(mat, beam)] = st
            lowest_load_by_group[(mat, beam)] = lv

    # Optional: Virtual compression of real lowest-load scans via DDIM ODE inversion
    virtual_cache_by_state: dict[str, np.ndarray] = {}
    vc_same_load_by_group: dict[tuple[str, str], np.ndarray] = {}
    vc_extrap_by_group: dict[tuple[str, str], dict[str, object]] = {}
    if args.include_virtual_compression:
        if torch.cuda.is_available():
            vc_device = torch.device("cuda")
        elif args.allow_cpu:
            vc_device = torch.device("cpu")
        else:
            raise RuntimeError("CUDA GPU required for --include-virtual-compression (or pass --allow-cpu)")

        ckpt_path = resolve_checkpoint_path(args.output_root, first_exp, args.checkpoint)
        if ckpt_path is None:
            raise FileNotFoundError(
                f"Checkpoint not found for virtual compression: experiment={first_exp!r}, requested={args.checkpoint}"
            )

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
            use_ordinal_material_channel=infer_use_ordinal_material_channel(saved_ckpt),
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
            target_lvs = list(
                dict.fromkeys([ref_lv, *[float(sp["load_value"]) for sp in grp_specs]])  # type: ignore[arg-type]
            )
            vc_extrap_by_group[(mat, beam)] = check_load_extrapolation(
                target_lvs, float(vc_cfg["load_min"]), float(vc_cfg["load_max"])
            )
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
            vc_same_load_by_group[(mat, beam)] = vc_map[ref_lv]
            for sp in grp_specs:
                st_k = str(sp["state"])
                lv_k = float(sp["load_value"])  # type: ignore[arg-type]
                virtual_cache_by_state[st_k] = vc_map[lv_k]

    max_lag = min(32, args.patch_size // 2)
    report: dict[str, object] = {}
    csv_rows: list[dict[str, object]] = []
    openimpala_entries: list[dict[str, object]] = []
    want_openimpala = (
        args.export_real_openimpala or args.generate_openimpala_suite or args.run_openimpala
    )

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
                        "voxel_size_um": vox,
                    }
                )

        real_recs = [evaluate_multiphase_physics(v, voxel_size=vox) for v in real_vols]
        real_sum = summarize_metric_records(real_recs)
        real_s2 = _mean_s2_curves(real_vols, max_lag=max_lag)
        target_sf = float(real_sum["solid_fraction"]["mean"])

        # Construct literature Z-compression baselines from tall clean uncompressed real sub-volumes
        ref_st = lowest_state_by_group[(mat, beam)]
        base_src_vols = tall_ref_cache[ref_st]
        phi_0_src = float((base_src_vols >= 1).mean())
        mirror_frac = compute_mirror_tiled_fraction(
            source_depth=int(base_src_vols.shape[1]),
            output_depth=args.patch_size,
            initial_solid_fraction=phi_0_src,
            target_solid_fraction=max(target_sf, 1e-3),
        )
        affine_vols = np.stack(
            [
                affine_z_compress_to_target(
                    rv, target_solid_fraction=max(target_sf, 1e-3), output_depth=args.patch_size
                )
                for rv in base_src_vols
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
                for i, rv in enumerate(base_src_vols)
            ],
            axis=0,
        )

        methods_vols: dict[str, np.ndarray] = {
            "noop_uncompressed": real_cache[ref_st],
            "affine_z_squash": affine_vols,
            "pillar_pore_deletion": pillar_vols,
        }
        if st in virtual_cache_by_state:
            methods_vols["virtual_compression"] = virtual_cache_by_state[st]

        per_seed_vols_by_method: dict[str, list[np.ndarray]] = {}
        provenance_by_method: dict[str, dict[str, object]] = {}
        for lbl, exp_list in exp_map.items():
            seed_vols: list[np.ndarray] = []
            weight_sources: list[str] = []
            for exp_name in exp_list:
                samples_dir = args.output_root / "results" / exp_name / "samples"
                if (samples_dir / st).is_dir():
                    seed_vols.append(_load_generated_volumes(samples_dir, st))
                prov_path = samples_dir / "sampling_provenance.json"
                if prov_path.is_file():
                    prov = json.loads(prov_path.read_text())
                    weight_sources.append(str(prov.get("weight_source", "unknown")))
            if seed_vols:
                methods_vols[lbl] = np.concatenate(seed_vols, axis=0)
                per_seed_vols_by_method[lbl] = seed_vols
                provenance_by_method[lbl] = {
                    "experiments": exp_list,
                    "num_seeds": len(seed_vols),
                    "weight_sources": weight_sources,
                    "used_non_ema_debug_weights": any(ws == "raw" for ws in weight_sources),
                }

        if want_openimpala:
            for m_name, m_vols in methods_vols.items():
                if m_name == "noop_uncompressed":
                    continue  # Identical to real_cache[ref_st], already exported under openimpala_real/
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
                            "voxel_size_um": vox,
                        }
                    )

        noop_s2_err = _s2_rmse(real_s2, _mean_s2_curves(methods_vols["noop_uncompressed"], max_lag=max_lag))
        methods_summary: dict[str, object] = {}
        for m_name, m_vols in methods_vols.items():
            m_recs = [evaluate_multiphase_physics(v, voxel_size=vox) for v in m_vols]
            m_sum = summarize_metric_records(m_recs)
            dist_d = compute_ensemble_distribution_distances(real_recs, m_recs, keys=key_metrics)

            seed_vols_list = per_seed_vols_by_method.get(m_name, [])
            per_seed_s2: list[float] = []
            if len(seed_vols_list) >= 2:
                per_seed_sums = []
                for sv in seed_vols_list:
                    sv_recs = [evaluate_multiphase_physics(v, voxel_size=vox) for v in sv]
                    per_seed_sums.append(summarize_metric_records(sv_recs))
                    per_seed_s2.append(_s2_rmse(real_s2, _mean_s2_curves(sv, max_lag=max_lag)))
                s2_err = float(np.mean(per_seed_s2))
                s2_std = float(np.std(per_seed_s2, ddof=1))
                for mk, md in m_sum.items():
                    s_means = [ps[mk]["mean"] for ps in per_seed_sums if np.isfinite(ps[mk]["mean"])]
                    md["seed_std"] = float(np.std(s_means, ddof=1)) if len(s_means) >= 2 else 0.0
            else:
                m_s2 = _mean_s2_curves(m_vols, max_lag=max_lag)
                s2_err = _s2_rmse(real_s2, m_s2)
                s2_std = 0.0

            s2_gain_vs_noop = round(noop_s2_err - s2_err, 6)
            m_entry: dict[str, object] = {
                "num_volumes": int(m_vols.shape[0]),
                "num_seeds": max(1, len(seed_vols_list)),
                "s2_rmse": s2_err,
                "s2_rmse_std": s2_std,
                "s2_improvement_over_noop": s2_gain_vs_noop,
                "metrics": m_sum,
                "distribution_distances": dist_d,
            }
            if per_seed_s2:
                m_entry["per_seed_s2_rmse"] = [round(v, 6) for v in per_seed_s2]
            if m_name in provenance_by_method:
                m_entry["provenance"] = provenance_by_method[m_name]
            methods_summary[m_name] = m_entry

            row: dict[str, object] = {
                "state": st,
                "material": mat,
                "beamline": beam,
                "load_value": lv,
                "load_unit": unit_label,
                "is_heldout": is_heldout,
                "method": m_name,
                "num_seeds": max(1, len(seed_vols_list)),
                "s2_rmse": round(s2_err, 6),
                "s2_rmse_std": round(s2_std, 6),
                "s2_improvement_over_noop": s2_gain_vs_noop,
                "baseline_mirror_tiled_fraction": (
                    round(mirror_frac, 6)
                    if m_name in ("affine_z_squash", "pillar_pore_deletion")
                    else 0.0
                ),
            }
            for km in key_metrics:
                r_mu = real_sum[km]["mean"]
                m_mu = m_sum[km]["mean"]
                w1_val = dist_d.get(km, {}).get("wasserstein_1", float("nan"))
                row[f"real_{km}"] = round(r_mu, 6) if np.isfinite(r_mu) else float("nan")
                row[f"pred_{km}"] = round(m_mu, 6) if np.isfinite(m_mu) else float("nan")
                row[f"abs_err_{km}"] = (
                    round(abs(m_mu - r_mu), 6)
                    if (np.isfinite(r_mu) and np.isfinite(m_mu))
                    else float("nan")
                )
                row[f"w1_{km}"] = round(w1_val, 6) if np.isfinite(w1_val) else float("nan")
            csv_rows.append(row)

        st_entry: dict[str, object] = {
            "material": mat,
            "beamline": beam,
            "load_value": lv,
            "load_unit": unit_label,
            "voxel_size_um": vox,
            "is_heldout": is_heldout,
            "reference_uncompressed_state": ref_st,
            "baseline_source_depth": int(base_src_vols.shape[1]),
            "baseline_mirror_tiled_fraction": round(mirror_frac, 6),
            "baseline_oracle_porosity_note": (
                "Affine and pillar baselines receive the held-out target solid_fraction by construction; "
                "noop_uncompressed evaluates the unmodified lowest-load scan directly against each target state. "
                "Evaluate morphological fidelity via S2(r), chord anisotropy, EASA/TPB, and tortuosity."
            ),
            "real": {"num_volumes": int(real_vols.shape[0]), "metrics": real_sum},
            "methods": methods_summary,
        }
        if st in virtual_cache_by_state:
            vc_diag = dict(
                compute_virtual_compression_fidelity(
                    source_volumes=real_cache[ref_st],
                    compressed_volumes=virtual_cache_by_state[st],
                    same_load_reconstructions=vc_same_load_by_group.get((mat, beam)),
                    is_target_same_as_source=(st == ref_st),
                )
            )
            vc_diag.update(vc_extrap_by_group.get((mat, beam), {}))
            vc_diag["noop_s2_rmse"] = round(noop_s2_err, 6)
            vc_diag["s2_improvement_over_noop"] = methods_summary["virtual_compression"][  # type: ignore[index]
                "s2_improvement_over_noop"
            ]
            st_entry["virtual_compression_diagnostics"] = vc_diag
        report[st] = st_entry
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
            template_ini=args.openimpala_template_ini,
            include_hetero_solid=True,
        )
        print(
            f"Generated OpenImpala batch suite ({suite_info['num_jobs']} jobs) -> {suite_info['script_path']}",
            flush=True,
        )
        if args.run_openimpala:
            if not has_openimpala_python():
                raise RuntimeError(
                    "--run-openimpala requested, but the 'openimpala' Python package is not installed."
                )
            oi_rows = run_openimpala_python_suite(
                tiff_entries=openimpala_entries,
                suite_dir=out_dir / "openimpala_suite",
                out_csv=out_dir / "openimpala_transport_parity.csv",
                include_hetero_solid=True,
            )
            print(
                f"Executed OpenImpala v4.6 in-process ({len(oi_rows)} phase solves) -> "
                f"{out_dir / 'openimpala_transport_parity.csv'}",
                flush=True,
            )
        elif (out_dir / "openimpala_suite" / "outputs").is_dir():
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
            palette = ["tab:purple", "tab:gray", "tab:orange", "tab:blue", "tab:green", "tab:red"]
            for m_i, m_name in enumerate(all_methods):
                my = [
                    report[st]["methods"][m_name]["metrics"][m_key]["mean"]  # type: ignore[index]
                    for st in mat_states
                    if m_name in report[st]["methods"]  # type: ignore[index]
                ]
                my_err = [
                    float(report[st]["methods"][m_name]["metrics"][m_key].get("seed_std", 0.0))  # type: ignore[index]
                    for st in mat_states
                    if m_name in report[st]["methods"]  # type: ignore[index]
                ]
                mx = [
                    float(report[st]["load_value"])  # type: ignore[index]
                    for st in mat_states
                    if m_name in report[st]["methods"]  # type: ignore[index]
                ]
                is_base = m_name in ("noop_uncompressed", "affine_z_squash", "pillar_pore_deletion")
                fmt = ":d" if m_name == "noop_uncompressed" else ("--s" if is_base else "-^")
                col = palette[m_i % len(palette)]
                if any(e > 0.0 for e in my_err):
                    ax.errorbar(mx, my, yerr=my_err, fmt=fmt, color=col, lw=1.5, capsize=3, label=m_name)
                else:
                    ax.plot(mx, my, fmt, color=col, lw=1.5, label=m_name)

            ax.set_title(f"{mat} — {m_label}", fontsize=10)
            ax.set_xlabel(f"Compressive Load / Stress ({unit_label})")
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
