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

from porousdiff.baselines import affine_z_compress_to_target, pillar_pore_deletion_compress
from porousdiff.data import default_project_root, load_split_patches, resolve_patch_split_dir
from porousdiff.metrics import (
    directional_two_point_correlation,
    evaluate_multiphase_physics,
    export_openimpala_tiff,
    summarize_metric_records,
)
from porousdiff.preprocessing import infer_material_name, parse_load_value


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
    parser.add_argument("--max-real-samples", type=int, default=20)
    parser.add_argument("--voxel-size-um", type=float, default=1.0)
    parser.add_argument("--export-real-openimpala", action="store_true")
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

    # Build state specification list grouped by material
    state_specs: list[dict[str, object]] = []
    if args.manifest_json is not None:
        manifest = json.loads(args.manifest_json.read_text())
        for sc in manifest.get("scans", []):
            if bool(sc.get("is_repeat_or_secondary", False)):
                continue
            state_specs.append(
                {
                    "state": str(sc["state"]),
                    "material": str(sc.get("material", "virgin_carbon")),
                    "load_value": float(sc.get("stress_kpa") or sc["load_mv_per_v"]),
                    "voxel_size_um": float(sc.get("effective_voxel_size_um", args.voxel_size_um)),
                    "num_phases": int(sc.get("num_phases", 2)),
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
                    "load_value": lv,
                    "voxel_size_um": args.voxel_size_um,
                    "num_phases": 3 if infer_material_name(st) != "virgin_carbon" else 2,
                }
            )
    else:
        raise ValueError("Provide either --manifest-json or --states")

    state_specs.sort(key=lambda s: (str(s["material"]), float(s["load_value"])))  # type: ignore[arg-type]

    # Identify lowest-load reference real validation patches per material for Z-compression baselines
    rng = np.random.default_rng(args.seed)
    expected_shape = (args.patch_size, args.patch_size, args.patch_size)
    real_cache: dict[str, np.ndarray] = {}
    lowest_state_by_mat: dict[str, str] = {}

    for spec in state_specs:
        st = str(spec["state"])
        mat = str(spec["material"])
        split_dir = resolve_patch_split_dir(args.data_root, st, args.patch_size, args.version, "val")
        vols = load_split_patches(split_dir, expected_shape=expected_shape)
        if vols.shape[0] > args.max_real_samples:
            idx = np.sort(rng.choice(vols.shape[0], size=args.max_real_samples, replace=False))
            vols = vols[idx]
        real_cache[st] = vols
        if mat not in lowest_state_by_mat:
            lowest_state_by_mat[mat] = st

    max_lag = min(32, args.patch_size // 2)
    report: dict[str, object] = {}
    csv_rows: list[dict[str, object]] = []

    key_metrics = [
        "solid_fraction",
        "coating_fraction",
        "ssa_pore_coating",
        "tpb_density",
        "wired_coating_fraction",
        "pore_anisotropy_xy_over_z",
        "geodesic_tortuosity_z",
    ]

    for spec in state_specs:
        st = str(spec["state"])
        mat = str(spec["material"])
        lv = float(spec["load_value"])  # type: ignore[arg-type]
        vox = float(spec["voxel_size_um"])  # type: ignore[arg-type]
        n_ph = int(spec["num_phases"])  # type: ignore[arg-type]
        is_heldout = st in heldout_set

        real_vols = real_cache[st]
        if args.export_real_openimpala:
            for r_i, rv in enumerate(real_vols[:5], start=1):
                export_openimpala_tiff(
                    rv,
                    out_dir / "openimpala_real" / st / f"real_val_{r_i:03d}.tif",
                    preserve_multiphase=(n_ph == 3),
                )

        real_recs = [evaluate_multiphase_physics(v, voxel_size=vox) for v in real_vols]
        real_sum = summarize_metric_records(real_recs)
        real_s2 = _mean_s2_curves(real_vols, max_lag=max_lag)
        target_sf = float(real_sum["solid_fraction"]["mean"])

        # Construct literature Z-compression baselines from uncompressed scan of same material
        ref_st = lowest_state_by_mat[mat]
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
        for lbl, exp_name in exp_map.items():
            samples_dir = args.output_root / "results" / exp_name / "samples"
            if (samples_dir / st).is_dir():
                methods_vols[lbl] = _load_generated_volumes(samples_dir, st)

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
            "load_value": lv,
            "voxel_size_um": vox,
            "is_heldout": is_heldout,
            "reference_uncompressed_state": ref_st,
            "real": {"num_volumes": int(real_vols.shape[0]), "metrics": real_sum},
            "methods": methods_summary,
        }
        print(
            f"[{st} | {mat} | heldout={is_heldout}] real φ_s={target_sf:.4f} | "
            + " | ".join(
                f"{m}: s2_rmse={methods_summary[m]['s2_rmse']:.4f}"  # type: ignore[index]
                for m in methods_summary
            ),
            flush=True,
        )

    with (out_dir / "beamtime_multiplier_report.json").open("w") as f:
        json.dump(report, f, indent=2)

    if csv_rows:
        with (out_dir / "baseline_vs_porousdiff_table.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

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
