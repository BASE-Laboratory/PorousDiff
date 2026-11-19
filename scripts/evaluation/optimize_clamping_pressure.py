"""Find optimal electrode clamping pressure `sigma*` coupling PorousDiff & OpenImpala transport.

Computes the continuous reaction-transport Figure of Merit `FoM(sigma)` and bootstrap 95%
confidence interval on the optimal cell clamping stress `sigma*` across electrode chemistries
(`virgin_carbon`, `tio2_anode`, `cuhcf_cathode`), comparing:
  - Real synchrotron XCT ground truth (`real`),
  - PorousDiff generative predictions (`porousdiff`, plus optional dense continuous stress sweep),
  - Standard literature numerical compression heuristics (`affine_z_squash`, `pillar_pore_deletion`).

Automatically ingests OpenImpala continuum solver outputs (`openimpala_transport_parity.csv`)
when present, and seamlessly falls back to exact 3D geodesic pore/solid tortuosity and
Carman-Kozeny permeability surrogates prior to running OpenImpala.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from porousdiff.baselines import affine_z_compress_to_target, pillar_pore_deletion_compress
from porousdiff.data import (
    decode_channels_to_phases,
    default_project_root,
    load_split_patches,
    load_tall_or_split_patches,
    resolve_checkpoint_path,
    resolve_material_id,
    resolve_patch_split_dir,
    resolve_scan_load_value,
    resolve_use_stress_kpa_from_checkpoints,
)
from porousdiff.metrics import evaluate_multiphase_physics
from porousdiff.models import (
    ContinuousLoadConditionedUNet,
    EMAModel,
    build_noise_scheduler,
    infer_use_ordinal_material_channel,
)
from porousdiff.optimization import (
    compute_transport_and_fom_record,
    find_optimal_clamping_pressure,
    load_openimpala_records,
)
from porousdiff.preprocessing import infer_material_name, parse_load_value
from porousdiff.transport_and_xrd import (
    compute_openimpala_volume_transport,
    has_openimpala_python,
)


def _parse_experiments(entries: list[str]) -> dict[str, list[str]]:
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


def _load_generated_volumes(samples_dir: Path, state_name: str) -> np.ndarray:
    state_dir = samples_dir / state_name
    stacked = state_dir / "generated_samples.npy"
    if stacked.is_file():
        return np.load(stacked)
    files = sorted(state_dir.glob("sample_*_binary.npy"))
    if not files:
        raise FileNotFoundError(f"No generated samples found in {state_dir}")
    return np.stack([np.load(p) for p in files], axis=0)


def _pair_volumes_with_openimpala(
    vols: Sequence[np.ndarray] | np.ndarray,
    oi_recs: Sequence[dict[str, float]] | None,
    voxel_size_um: float,
    strain_z: float | None,
    solve_in_process: bool = False,
) -> list[dict[str, float]]:
    """Pair each 3D volume 1-to-1 with its matching OpenImpala PDE record (never broadcasting a group mean)."""
    vol_list = list(vols)
    if oi_recs is None and not solve_in_process:
        return [
            compute_transport_and_fom_record(
                evaluate_multiphase_physics(v, voxel_size=voxel_size_um),
                openimpala_metrics=None,
                voxel_size_um=voxel_size_um,
                strain_z=strain_z,
            )
            for v in vol_list
        ]

    paired: list[dict[str, float]] = []
    cached = list(oi_recs) if oi_recs is not None else []
    can_solve = solve_in_process and has_openimpala_python()
    for idx, v in enumerate(vol_list):
        if idx < len(cached):
            oi_m = cached[idx]
        elif can_solve:
            oi_m = compute_openimpala_volume_transport(v, voxel_size_um=voxel_size_um)["lookup_summary"]  # type: ignore[assignment]
        else:
            break
        paired.append(
            compute_transport_and_fom_record(
                evaluate_multiphase_physics(v, voxel_size=voxel_size_um),
                openimpala_metrics=oi_m,
                voxel_size_um=voxel_size_um,
                strain_z=strain_z,
            )
        )
    return paired


@torch.inference_mode()
def _sample_dense_sweep_for_material(
    active_model: ContinuousLoadConditionedUNet,
    scheduler: object,
    dense_loads: np.ndarray,
    num_samples_per_load: int,
    patch_size: int,
    guidance_scale: float,
    material_id: int,
    voxel_size_um: float,
    seed: int,
    use_amp: bool,
    amp_dtype: torch.dtype,
    use_openimpala: bool = False,
) -> dict[float, list[dict[str, float]]]:
    """Generate a continuous stress sweep with shared latent seeds to resolve `sigma*` finely."""
    device = next(active_model.parameters()).device
    data_ch = active_model.data_channels
    gen = torch.Generator(device=device).manual_seed(seed)
    shared_noise = torch.randn(
        (num_samples_per_load, data_ch, patch_size, patch_size, patch_size),
        generator=gen,
        device=device,
    )
    records_by_load: dict[float, list[dict[str, float]]] = {}
    for lv in dense_loads:
        lv_f = float(lv)
        img = shared_noise.clone()
        loads_t = torch.full((num_samples_per_load,), lv_f, device=device, dtype=torch.float32)
        mats_t = (
            torch.full((num_samples_per_load,), int(material_id), device=device, dtype=torch.long)
            if active_model.is_multimodal
            else None
        )
        voxs_t = (
            torch.full(
                (num_samples_per_load,), float(voxel_size_um), device=device, dtype=torch.float32
            )
            if active_model.is_multimodal
            else None
        )
        step_gen = torch.Generator(device=device).manual_seed(seed + 101)
        for timestep in scheduler.timesteps:  # type: ignore[attr-defined]
            t_int = int(timestep)
            t_batch = torch.full((num_samples_per_load,), t_int, device=device, dtype=torch.long)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                pred = active_model.forward_cfg(
                    img,
                    t_batch,
                    loads_t,
                    guidance_scale=guidance_scale,
                    material_ids=mats_t,
                    voxel_sizes=voxs_t,
                )
            step_out = scheduler.step(pred.float(), t_int, img, generator=step_gen)[0]  # type: ignore[attr-defined]
            img = torch.nan_to_num(step_out, nan=0.0, posinf=1.0, neginf=-1.0)

        vols = decode_channels_to_phases(img.float().cpu().numpy(), enforce_solid_support=True)
        recs: list[dict[str, float]] = []
        for v in vols:
            oi_m = (
                compute_openimpala_volume_transport(v, voxel_size_um=voxel_size_um)["lookup_summary"]  # type: ignore[assignment]
                if use_openimpala
                else None
            )
            recs.append(
                compute_transport_and_fom_record(
                    evaluate_multiphase_physics(v, voxel_size=voxel_size_um),
                    openimpala_metrics=oi_m,
                    voxel_size_um=voxel_size_um,
                )
            )
        records_by_load[lv_f] = recs
    return records_by_load


def main() -> None:
    proj_root = default_project_root()
    parser = argparse.ArgumentParser(
        description="Discover optimal electrode clamping pressure sigma* across Real XCT, OpenImpala, and PorousDiff"
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        required=True,
        help="Experiment(s), e.g. porousdiff=beamtime_multiplier_96",
    )
    parser.add_argument("--data-root", type=Path, default=None, help="Optional real XCT patch root for ground-truth & baselines")
    parser.add_argument("--output-root", type=Path, default=proj_root)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--manifest-json", type=Path, default=None)
    parser.add_argument("--states", nargs="*", default=None)
    parser.add_argument("--version", default="carbon_calibrated_v1")
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--voxel-size-um", type=float, default=1.0)
    parser.add_argument("--include-repeats", action="store_true")
    parser.add_argument("--beamline", default=None)
    parser.add_argument(
        "--use-stress-kpa",
        action="store_true",
        help="Fallback load unit when no checkpoint exists under --output-root (otherwise read from checkpoint config)",
    )
    parser.add_argument("--max-real-samples", type=int, default=20)
    parser.add_argument(
        "--openimpala-csv",
        type=Path,
        default=None,
        help="Optional openimpala_transport_parity.csv from parse_openimpala_suite.py / --run-openimpala",
    )
    parser.add_argument(
        "--run-openimpala",
        action="store_true",
        help="Solve OpenImpala v4.6 PDE in-process across uncached realizations and dense sweep samples",
    )
    parser.add_argument(
        "--allow-mixed-dense-estimator",
        action="store_true",
        help="Allow geodesic-surrogate porousdiff_dense curve alongside OpenImpala discrete curves when openimpala is uninstalled",
    )
    parser.add_argument(
        "--weights",
        default="0.35,0.35,0.30",
        help="Comma-separated weights 'ionic,electronic,kinetic' for harmonic cell conductance FoM",
    )
    parser.add_argument("--dense-sweep-points", type=int, default=0, help=">0 runs continuous intermediate PorousDiff stress sweep")
    parser.add_argument("--dense-samples-per-load", type=int, default=4)
    parser.add_argument(
        "--uncompressed-thickness-cm",
        type=float,
        default=0.30,
        help="Uncompressed felt thickness L_0 in cm (default 0.30 cm = 3.0 mm) for 1D Newman ASR",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=1.5)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    w_parts = [float(x.strip()) for x in args.weights.split(",") if x.strip()]
    if len(w_parts) != 3:
        raise ValueError("--weights must have 3 comma-separated floats: ionic,electronic,kinetic")
    weights = (w_parts[0], w_parts[1], w_parts[2])

    exp_map = _parse_experiments(args.experiments)
    all_exp_names = [e for exps in exp_map.values() for e in exps]
    first_exp = all_exp_names[0]
    out_dir = (
        args.out_dir
        if args.out_dir is not None
        else (args.output_root / "results" / first_exp / "clamping_optimization")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    use_kpa, _ = resolve_use_stress_kpa_from_checkpoints(
        output_root=args.output_root,
        experiments=all_exp_names,
        explicit_checkpoint=args.checkpoint,
        default=args.use_stress_kpa,
    )

    auto_oi_csv = (
        args.openimpala_csv
        if args.openimpala_csv is not None
        else (args.output_root / "results" / first_exp / "beamtime_benchmark" / "openimpala_transport_parity.csv")
    )
    oi_lookup = load_openimpala_records(auto_oi_csv) if auto_oi_csv.is_file() else {}

    # Build state specs from manifest, --states, or sampled directories
    state_specs: list[dict[str, object]] = []
    if args.manifest_json is not None:
        manifest = json.loads(args.manifest_json.read_text())
        all_m_scans = list(manifest.get("scans", []))
        if all_m_scans and args.patch_size == 96 and "patch_size" in all_m_scans[0]:
            args.patch_size = int(all_m_scans[0]["patch_size"])
        for sc in all_m_scans:
            if not args.include_repeats and bool(sc.get("is_repeat_or_secondary", False)):
                continue
            sc_beam = str(sc.get("beamline", "primary"))
            if args.beamline and sc_beam != args.beamline:
                continue
            lv_resolved = resolve_scan_load_value(sc, use_kpa)
            raw_sz = sc.get("strain_z")
            state_specs.append(
                {
                    "state": str(sc["state"]),
                    "material": str(sc.get("material", "virgin_carbon")),
                    "beamline": sc_beam,
                    "load_value": lv_resolved,
                    "strain_z": float(raw_sz) if raw_sz is not None else None,
                    "voxel_size_um": float(sc.get("effective_voxel_size_um", args.voxel_size_um)),
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
                    "strain_z": None,
                    "voxel_size_um": args.voxel_size_um,
                    "patch_dir_name": None,
                }
            )
    else:
        # Discover from first experiment's per_sample_physics.csv
        sample_csv = args.output_root / "results" / first_exp / "samples" / "per_sample_physics.csv"
        if not sample_csv.is_file():
            raise ValueError("Provide --manifest-json, --states, or run sample_carbon_conditional_96.py first")
        seen_st: set[str] = set()
        with sample_csv.open("r", newline="") as f:
            for row in csv.DictReader(f):
                st = str(row["state"])
                if st in seen_st:
                    continue
                seen_st.add(st)
                state_specs.append(
                    {
                        "state": st,
                        "material": str(row.get("material", infer_material_name(st))),
                        "beamline": "primary",
                        "load_value": float(row["load_value"]),
                        "strain_z": None,
                        "voxel_size_um": float(row.get("voxel_size_um", args.voxel_size_um)),
                        "patch_dir_name": None,
                    }
                )

    state_specs.sort(key=lambda s: (str(s["material"]), float(s["load_value"])))  # type: ignore[arg-type]

    # Collect per-(material, method, load) transport records
    rng = np.random.default_rng(args.seed)
    expected_shape = (args.patch_size, args.patch_size, args.patch_size)
    records_tree: dict[str, dict[str, dict[float, list[dict[str, float]]]]] = {}

    manifest_fallback_root = args.manifest_json.parent if args.manifest_json is not None else None
    have_real_data = (args.data_root is not None and args.data_root.is_dir()) or (
        manifest_fallback_root is not None and manifest_fallback_root.is_dir()
    )
    real_cache: dict[str, np.ndarray] = {}
    tall_real_cache: dict[str, np.ndarray] = {}
    lowest_state_by_mat: dict[str, str] = {}

    if have_real_data:
        eff_root = args.data_root if args.data_root is not None else manifest_fallback_root
        assert eff_root is not None
        for sp in state_specs:
            st = str(sp["state"])
            mat = str(sp["material"])
            pdir_hint = str(sp["patch_dir_name"]) if sp.get("patch_dir_name") else None
            try:
                split_dir = resolve_patch_split_dir(
                    eff_root,
                    st,
                    args.patch_size,
                    args.version,
                    "val",
                    patch_dir_name=pdir_hint,
                    fallback_root=manifest_fallback_root,
                )
                vols = load_split_patches(split_dir, expected_shape=expected_shape)
            except (FileNotFoundError, RuntimeError):
                continue
            if vols.shape[0] > args.max_real_samples:
                idx = np.sort(rng.choice(vols.shape[0], size=args.max_real_samples, replace=False))
                vols = vols[idx]
            real_cache[st] = vols
            tall_vols = load_tall_or_split_patches(split_dir, vols)
            if tall_vols.shape[0] > args.max_real_samples:
                tall_vols = tall_vols[: args.max_real_samples]
            tall_real_cache[st] = tall_vols
            if mat not in lowest_state_by_mat:
                lowest_state_by_mat[mat] = st

    # Verify per-(material, method) OpenImpala coverage so curves never mix partial OpenImpala + surrogate
    oi_method_complete: dict[tuple[str, str], bool] = {}
    if oi_lookup:
        mat_to_real_states: dict[str, list[str]] = {}
        mat_to_exp_states: dict[tuple[str, str], list[str]] = {}
        for sp in state_specs:
            st = str(sp["state"])
            mat = str(sp["material"])
            if st in real_cache:
                mat_to_real_states.setdefault(mat, []).append(st)
            for lbl, exp_list in exp_map.items():
                if any((args.output_root / "results" / exp_n / "samples" / st).is_dir() for exp_n in exp_list):
                    mat_to_exp_states.setdefault((mat, lbl), []).append(st)

        def _check_oi_complete(mat_k: str, meth_k: str, states_list: list[str]) -> None:
            if not states_list:
                return
            cov = sum(1 for st_k in states_list if (meth_k, st_k) in oi_lookup)
            if cov == len(states_list):
                oi_method_complete[(mat_k, meth_k)] = True
            else:
                oi_method_complete[(mat_k, meth_k)] = False
                if cov > 0:
                    print(
                        f"Warning: OpenImpala results cover only {cov}/{len(states_list)} states for "
                        f"({mat_k}, {meth_k}); falling back to uniform 26-connected geodesic estimator "
                        "across all loads to avoid mixing two transport estimators.",
                        flush=True,
                    )

        for mat_k, r_sts in mat_to_real_states.items():
            for meth_k in ("real", "affine_z_squash", "pillar_pore_deletion"):
                _check_oi_complete(mat_k, meth_k, r_sts)
        for (mat_k, lbl_k), g_sts in mat_to_exp_states.items():
            _check_oi_complete(mat_k, lbl_k, g_sts)

    per_seed_records_tree: dict[str, dict[str, dict[str, dict[float, list[dict[str, float]]]]]] = {}
    provenance_by_method: dict[str, dict[str, object]] = {}
    for lbl, exp_list in exp_map.items():
        ws_list: list[str] = []
        for exp_name in exp_list:
            prov_p = args.output_root / "results" / exp_name / "samples" / "sampling_provenance.json"
            if prov_p.is_file():
                prov = json.loads(prov_p.read_text())
                ws_list.append(str(prov.get("weight_source", "unknown")))
        provenance_by_method[lbl] = {
            "experiments": exp_list,
            "weight_sources": ws_list,
            "used_non_ema_debug_weights": any(ws == "raw" for ws in ws_list),
        }

    for sp in state_specs:
        st = str(sp["state"])
        mat = str(sp["material"])
        lv = float(sp["load_value"])  # type: ignore[arg-type]
        vox = float(sp["voxel_size_um"])  # type: ignore[arg-type]
        sz_opt = float(sp["strain_z"]) if sp.get("strain_z") is not None else None  # type: ignore[arg-type]
        mat_bucket = records_tree.setdefault(mat, {})

        if st in real_cache:
            r_vols = real_cache[st]
            r_oi = oi_lookup.get(("real", st)) if oi_method_complete.get((mat, "real"), False) else None
            r_recs = _pair_volumes_with_openimpala(
                r_vols,
                r_oi,
                voxel_size_um=vox,
                strain_z=sz_opt,
                solve_in_process=args.run_openimpala,
            )
            mat_bucket.setdefault("real", {}).setdefault(lv, []).extend(r_recs)

            ref_st_mat = lowest_state_by_mat[mat]
            noop_vols = real_cache[ref_st_mat]
            noop_oi = oi_lookup.get(("real", ref_st_mat)) if oi_method_complete.get((mat, "real"), False) else None
            mat_bucket.setdefault("noop_uncompressed", {}).setdefault(lv, []).extend(
                _pair_volumes_with_openimpala(
                    noop_vols,
                    noop_oi,
                    voxel_size_um=vox,
                    strain_z=sz_opt,
                    solve_in_process=args.run_openimpala,
                )
            )

            target_sf = float(np.mean([(v >= 1).mean() for v in r_vols]))
            ref_vols = tall_real_cache[ref_st_mat]
            aff_vols = [
                affine_z_compress_to_target(rv, target_solid_fraction=max(target_sf, 1e-3), output_depth=args.patch_size)
                for rv in ref_vols
            ]
            pil_vols = [
                pillar_pore_deletion_compress(
                    rv,
                    target_solid_fraction=max(target_sf, 1e-3),
                    output_depth=args.patch_size,
                    rng=np.random.default_rng(args.seed + i),
                )
                for i, rv in enumerate(ref_vols)
            ]
            aff_oi = (
                oi_lookup.get(("affine_z_squash", st))
                if oi_method_complete.get((mat, "affine_z_squash"), False)
                else None
            )
            pil_oi = (
                oi_lookup.get(("pillar_pore_deletion", st))
                if oi_method_complete.get((mat, "pillar_pore_deletion"), False)
                else None
            )
            mat_bucket.setdefault("affine_z_squash", {}).setdefault(lv, []).extend(
                _pair_volumes_with_openimpala(
                    aff_vols,
                    aff_oi,
                    voxel_size_um=vox,
                    strain_z=sz_opt,
                    solve_in_process=args.run_openimpala,
                )
            )
            mat_bucket.setdefault("pillar_pore_deletion", {}).setdefault(lv, []).extend(
                _pair_volumes_with_openimpala(
                    pil_vols,
                    pil_oi,
                    voxel_size_um=vox,
                    strain_z=sz_opt,
                    solve_in_process=args.run_openimpala,
                )
            )

        for lbl, exp_list in exp_map.items():
            g_oi_all = oi_lookup.get((lbl, st)) if oi_method_complete.get((mat, lbl), False) else None
            oi_offset = 0
            for exp_name in exp_list:
                samples_dir = args.output_root / "results" / exp_name / "samples"
                if (samples_dir / st).is_dir():
                    g_vols = _load_generated_volumes(samples_dir, st)
                    seed_oi = (
                        g_oi_all[oi_offset : oi_offset + int(g_vols.shape[0])]
                        if g_oi_all is not None
                        else None
                    )
                    oi_offset += int(g_vols.shape[0])
                    g_recs = _pair_volumes_with_openimpala(
                        g_vols,
                        seed_oi,
                        voxel_size_um=vox,
                        strain_z=sz_opt,
                        solve_in_process=args.run_openimpala,
                    )
                    if g_recs:
                        mat_bucket.setdefault(lbl, {}).setdefault(lv, []).extend(g_recs)
                        (
                            per_seed_records_tree.setdefault(mat, {})
                            .setdefault(lbl, {})
                            .setdefault(exp_name, {})
                            .setdefault(lv, [])
                            .extend(g_recs)
                        )

    # Optional continuous dense PorousDiff sweep for high-resolution sigma* discovery
    if args.dense_sweep_points >= 3:
        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else (torch.device("cpu") if args.allow_cpu else None)
        )
        if device is None:
            raise RuntimeError("CUDA required for --dense-sweep-points (or pass --allow-cpu)")
        ckpt_path = resolve_checkpoint_path(
            output_root=args.output_root,
            experiment=first_exp,
            explicit_checkpoint=args.checkpoint,
        )
        if ckpt_path is None:
            raise FileNotFoundError(f"Checkpoint not found for experiment {first_exp!r} under {args.output_root}")
        saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = saved["config"]
        num_materials = int(cfg.get("num_materials", 1))
        model = ContinuousLoadConditionedUNet(
            channels=cfg["channels"],
            attention_levels=cfg["attention_levels"],
            num_res_blocks=int(cfg["num_res_blocks"]),
            norm_num_groups=int(cfg["norm_groups"]),
            load_min=float(cfg["load_min"]),
            load_max=float(cfg["load_max"]),
            use_flash_attention=(device.type == "cuda"),
            conditioning_mode=str(cfg.get("conditioning_mode", "multiscale_fourier")),
            num_phases=int(cfg.get("num_phases", 2)),
            num_materials=num_materials,
            ref_voxel_size_um=float(cfg.get("ref_voxel_size_um", 1.0)),
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
        sched = build_noise_scheduler(
            sampler="ddim",
            num_train_timesteps=int(cfg["timesteps"]),
            schedule=str(cfg["schedule"]),
            prediction_type=str(cfg["prediction_type"]),
        )
        sched.set_timesteps(args.steps, device=device)
        use_amp = device.type == "cuda"
        amp_dtype = (
            torch.bfloat16
            if (device.type == "cuda" and torch.cuda.is_bf16_supported())
            else torch.float16
        )

        for mat, mat_bucket in records_tree.items():
            mat_loads = sorted({float(sp["load_value"]) for sp in state_specs if str(sp["material"]) == mat})  # type: ignore[arg-type]
            if len(mat_loads) < 2:
                continue
            mat_uses_oi = args.run_openimpala or any(
                oi_method_complete.get((mat, m_k), False)
                for m_k in ("real", "affine_z_squash", "pillar_pore_deletion", *exp_map.keys())
            )
            dense_oi = False
            if mat_uses_oi:
                if has_openimpala_python():
                    dense_oi = True
                elif not args.allow_mixed_dense_estimator:
                    print(
                        f"Warning: Skipping porousdiff_dense for {mat} because discrete curves use OpenImpala PDE "
                        "while openimpala Python package is not installed (pass --allow-mixed-dense-estimator to override).",
                        flush=True,
                    )
                    continue
            dense_grid = np.linspace(mat_loads[0], mat_loads[-1], args.dense_sweep_points)
            vox_um = float(
                next(sp["voxel_size_um"] for sp in state_specs if str(sp["material"]) == mat)  # type: ignore[arg-type]
            )
            mat_id = resolve_material_id(mat) if num_materials > 1 else 0
            mat_bucket["porousdiff_dense"] = _sample_dense_sweep_for_material(
                active_model=active_model,
                scheduler=sched,
                dense_loads=dense_grid,
                num_samples_per_load=args.dense_samples_per_load,
                patch_size=int(cfg["patch_size"]),
                guidance_scale=args.guidance_scale,
                material_id=mat_id,
                voxel_size_um=vox_um,
                seed=args.seed,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                use_openimpala=dense_oi,
            )

    summary_out: dict[str, dict[str, object]] = {}
    csv_rows: list[dict[str, object]] = []

    for mat, methods_map in sorted(records_tree.items()):
        if not methods_map:
            continue
        mat_summary: dict[str, object] = {}
        ref_scales: tuple[float, float, float] | None = None
        if "real" in methods_map:
            real_opt = find_optimal_clamping_pressure(
                methods_map["real"],
                weights=weights,
                n_bootstrap=args.bootstrap,
                seed=args.seed,
                uncompressed_thickness_cm=args.uncompressed_thickness_cm,
            )
            ns = real_opt["norm_scales"]  # type: ignore[index]
            ref_scales = (
                float(ns["d_eff_pore_z"]),  # type: ignore[index]
                float(ns["sigma_eff_solid_z"]),  # type: ignore[index]
                float(ns["active_wired_interface"]),  # type: ignore[index]
            )
            mat_summary["real"] = real_opt

        real_sigma_star = (
            float(mat_summary["real"]["optimal_load"])  # type: ignore[index]
            if "real" in mat_summary
            else float("nan")
        )
        real_sigma_star_asr = (
            float(mat_summary["real"]["newman_asr"]["optimal_load_asr"])  # type: ignore[index]
            if "real" in mat_summary
            else float("nan")
        )

        for m_name, load_map in methods_map.items():
            if m_name == "real":
                opt_res = mat_summary["real"]  # type: ignore[assignment]
            else:
                opt_res = find_optimal_clamping_pressure(
                    load_map,
                    weights=weights,
                    n_bootstrap=args.bootstrap,
                    seed=args.seed,
                    norm_scales=ref_scales,
                    uncompressed_thickness_cm=args.uncompressed_thickness_cm,
                )
                if np.isfinite(real_sigma_star):
                    opt_res["abs_err_optimal_load_vs_real"] = round(
                        abs(float(opt_res["optimal_load"]) - real_sigma_star), 6
                    )
                if np.isfinite(real_sigma_star_asr):
                    opt_res["abs_err_optimal_load_asr_vs_real"] = round(
                        abs(float(opt_res["newman_asr"]["optimal_load_asr"]) - real_sigma_star_asr), 6  # type: ignore[index]
                    )
                seed_maps = per_seed_records_tree.get(mat, {}).get(m_name, {})
                valid_seed_maps = {s_k: s_m for s_k, s_m in seed_maps.items() if len(s_m) >= 2}
                if len(valid_seed_maps) >= 2:
                    s_opt_fom: list[float] = []
                    s_opt_asr: list[float] = []
                    for s_i, (_, s_lmap) in enumerate(valid_seed_maps.items()):
                        s_res = find_optimal_clamping_pressure(
                            s_lmap,
                            weights=weights,
                            n_bootstrap=max(10, args.bootstrap // 4),
                            seed=args.seed + s_i,
                            norm_scales=ref_scales,
                            uncompressed_thickness_cm=args.uncompressed_thickness_cm,
                        )
                        s_opt_fom.append(float(s_res["optimal_load"]))
                        s_opt_asr.append(float(s_res["newman_asr"]["optimal_load_asr"]))  # type: ignore[index]
                    opt_res["multi_seed"] = {
                        "num_seeds": len(valid_seed_maps),
                        "per_seed_optimal_load": [round(v, 6) for v in s_opt_fom],
                        "optimal_load_seed_mean": round(float(np.mean(s_opt_fom)), 6),
                        "optimal_load_seed_std": round(float(np.std(s_opt_fom, ddof=1)), 6),
                        "per_seed_optimal_load_asr": [round(v, 6) for v in s_opt_asr],
                        "optimal_load_asr_seed_mean": round(float(np.mean(s_opt_asr)), 6),
                        "optimal_load_asr_seed_std": round(float(np.std(s_opt_asr, ddof=1)), 6),
                    }
                if m_name in provenance_by_method:
                    opt_res["provenance"] = provenance_by_method[m_name]
                mat_summary[m_name] = opt_res

            for pt in opt_res["discrete_points"]:  # type: ignore[index]
                row: dict[str, object] = {
                    "material": mat,
                    "method": m_name,
                    "optimal_load_method": opt_res["optimal_load"],  # type: ignore[index]
                    "optimal_load_asr_method": opt_res["newman_asr"]["optimal_load_asr"],  # type: ignore[index]
                }
                row.update(pt)
                csv_rows.append(row)

            asr_info = opt_res["newman_asr"]  # type: ignore[index]
            print(
                f"[{mat} | {m_name}] sigma* (FoM) = {opt_res['optimal_load']} "  # type: ignore[index]
                f"(95% CI {opt_res['optimal_load_ci95']}) | "  # type: ignore[index]
                f"sigma* (Newman ASR) = {asr_info['optimal_load_asr']} "  # type: ignore[index]
                f"(min ASR = {asr_info['min_asr_ohm_cm2']} Ohm*cm^2, "  # type: ignore[index]
                f"grid halfwidth ±{asr_info['grid_spacing_halfwidth_asr']})",  # type: ignore[index]
                flush=True,
            )
        summary_out[mat] = mat_summary

    (out_dir / "optimal_clamping_summary.json").write_text(json.dumps(summary_out, indent=2))
    if csv_rows:
        with (out_dir / "clamping_optimization_curves.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    # Render 4-panel publication figure per material
    materials = list(summary_out.keys())
    if materials:
        fig, axes = plt.subplots(
            len(materials), 4, figsize=(16.5, 4.0 * len(materials)), squeeze=False
        )
        style_map = {
            "real": ("black", "-o", 2.2),
            "affine_z_squash": ("tab:gray", "--s", 1.4),
            "pillar_pore_deletion": ("tab:orange", "--d", 1.4),
        }
        extra_colors = ["tab:blue", "tab:green", "tab:purple", "tab:red"]

        for r_idx, mat in enumerate(materials):
            mat_res = summary_out[mat]
            extra_i = 0
            for m_name, m_obj in mat_res.items():
                pts = m_obj["discrete_points"]  # type: ignore[index]
                curve = m_obj["continuous_curve"]  # type: ignore[index]
                xs = [float(p["load_value"]) for p in pts]
                if m_name in style_map:
                    color, marker_style, lw = style_map[m_name]
                else:
                    color = extra_colors[extra_i % len(extra_colors)]
                    marker_style, lw = "-^", 2.0
                    extra_i += 1

                axes[r_idx, 0].plot(
                    xs, [float(p["d_eff_pore_z"]) for p in pts], marker_style, color=color, lw=lw, label=m_name
                )
                axes[r_idx, 1].plot(
                    xs, [float(p["sigma_eff_solid_z"]) for p in pts], marker_style, color=color, lw=lw, label=m_name
                )
                axes[r_idx, 2].plot(
                    xs, [float(p["active_wired_interface"]) for p in pts], marker_style, color=color, lw=lw, label=m_name
                )

                gx = np.asarray(curve["grid_loads"], dtype=np.float64)
                gfom = np.asarray(curve["fom_mean"], dtype=np.float64)
                glo = np.asarray(curve["fom_ci95_low"], dtype=np.float64)
                ghi = np.asarray(curve["fom_ci95_high"], dtype=np.float64)
                ls = "--" if m_name in ("affine_z_squash", "pillar_pore_deletion") else "-"
                axes[r_idx, 3].plot(gx, gfom, ls, color=color, lw=lw, label=f"{m_name} (σ*={m_obj['optimal_load']:g})")  # type: ignore[index]
                axes[r_idx, 3].fill_between(gx, glo, ghi, color=color, alpha=0.14)
                axes[r_idx, 3].axvline(float(m_obj["optimal_load"]), color=color, ls=":", lw=1.2)  # type: ignore[index]

            titles = [
                f"{mat} — Ionic D_eff,Z / D_0",
                f"{mat} — Electronic σ_eff,Z / σ_0",
                f"{mat} — Wired Active Interface (1/µm)",
                f"{mat} — Cell Conductance FoM & Optimal σ*",
            ]
            unit_lbl = "kPa" if use_kpa else "mV/V"
            for c_idx, ttl in enumerate(titles):
                axes[r_idx, c_idx].set_title(ttl, fontsize=10)
                axes[r_idx, c_idx].set_xlabel(f"Compressive Load / Stress ({unit_lbl})")
                axes[r_idx, c_idx].grid(True, alpha=0.3)
            axes[r_idx, 3].legend(fontsize=7.5, loc="best")
            if r_idx == 0:
                axes[r_idx, 0].legend(fontsize=7.5, loc="best")

        fig.suptitle(
            "Reaction-Transport Trade-Off & Optimal Cell Clamping Pressure σ* (OpenImpala + PorousDiff)",
            fontsize=13,
        )
        fig.tight_layout()
        fig.savefig(out_dir / "optimal_clamping_pressure.png", dpi=240, bbox_inches="tight")
        fig.savefig(out_dir / "optimal_clamping_pressure.pdf", bbox_inches="tight")
        plt.close(fig)

    print(f"Saved optimal clamping-pressure study -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
