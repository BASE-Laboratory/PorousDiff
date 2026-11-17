"""Quantitative 3D physical validation comparing real XCT patches vs generated volumes.

Evaluates both training loads and held-out interpolated loads across:
  - Solid fraction / porosity vs mechanical load
  - Specific surface area (total and directional)
  - Pore & fiber chord-length distributions and XY/Z compression anisotropy
  - Directional two-point correlation functions S_2(r_z), S_2(r_y), S_2(r_x)
  - 3D solid network connectivity, pore percolation, and geodesic tortuosity
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

from porousdiff.data import default_project_root, load_split_patches
from porousdiff.metrics import (
    directional_two_point_correlation,
    evaluate_volume_physics,
    summarize_metric_records,
)
from porousdiff.preprocessing import parse_load_value


def _load_generated_state_volumes(samples_dir: Path, state_name: str) -> np.ndarray:
    state_dir = samples_dir / state_name
    stacked = state_dir / "generated_samples.npy"
    if stacked.is_file():
        return np.load(stacked)
    files = sorted(state_dir.glob("*_binary.npy"))
    if not files:
        raise FileNotFoundError(f"No generated samples found in {state_dir}")
    return np.stack([np.load(p) for p in files], axis=0)


def main() -> None:
    proj_root = default_project_root()
    parser = argparse.ArgumentParser(
        description="Compare 3D physical & transport metrics of real vs generated microstructures"
    )
    parser.add_argument("--experiment", default="carbon_conditional_96_v2")
    parser.add_argument("--data-root", type=Path, default=proj_root / "data" / "carbon_virgin")
    parser.add_argument("--output-root", type=Path, default=proj_root)
    parser.add_argument("--version", default="carbon_calibrated_v1")
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--real-split", default="val", choices=["val", "train"])
    parser.add_argument("--max-real-samples", type=int, default=30)
    parser.add_argument("--voxel-size-um", type=float, default=1.0)
    parser.add_argument(
        "--states",
        nargs="+",
        required=True,
        help="States to compare (e.g. Op008mVV=0.008 Op018mVV=0.018 Op033mVV=0.033)",
    )
    parser.add_argument(
        "--heldout-states",
        nargs="*",
        default=[],
        help="Optional state names that were held out from training (highlighted on plots)",
    )
    args = parser.parse_args()

    states: dict[str, float] = {}
    for item in args.states:
        if "=" in item:
            k, v = item.split("=", 1)
            states[k.strip()] = float(v.strip())
        else:
            states[item.strip()] = parse_load_value(item.strip())

    heldout = set(args.heldout_states)
    exp_dir = args.output_root / "results" / args.experiment
    samples_dir = exp_dir / "samples"
    eval_dir = exp_dir / "physics_validation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    comparison_rows: list[dict[str, object]] = []
    full_report: dict[str, object] = {}
    s2_store: dict[str, dict[str, np.ndarray]] = {}

    for state_name, load_val in sorted(states.items(), key=lambda kv: kv[1]):
        real_split_dir = (
            args.data_root
            / f"{state_name}_binary_patches{args.patch_size}_{args.version}"
            / args.real_split
        )
        real_vols = load_split_patches(real_split_dir)
        if real_vols.shape[0] > args.max_real_samples:
            idx = rng.choice(real_vols.shape[0], size=args.max_real_samples, replace=False)
            real_vols = real_vols[np.sort(idx)]

        gen_vols = _load_generated_state_volumes(samples_dir, state_name)

        real_records = [evaluate_volume_physics(v, voxel_size=args.voxel_size_um) for v in real_vols]
        gen_records = [evaluate_volume_physics(v, voxel_size=args.voxel_size_um) for v in gen_vols]

        real_sum = summarize_metric_records(real_records)
        gen_sum = summarize_metric_records(gen_records)

        # Average directional S_2(r) curves
        real_s2_list = [directional_two_point_correlation(v) for v in real_vols[:12]]
        gen_s2_list = [directional_two_point_correlation(v) for v in gen_vols[:12]]
        lags = real_s2_list[0]["lags"]
        s2_entry = {
            "lags": lags,
            "real_s2_z": np.mean([c["s2_z"] for c in real_s2_list], axis=0),
            "gen_s2_z": np.mean([c["s2_z"] for c in gen_s2_list], axis=0),
            "real_s2_xy": 0.5 * (
                np.mean([c["s2_x"] for c in real_s2_list], axis=0)
                + np.mean([c["s2_y"] for c in real_s2_list], axis=0)
            ),
            "gen_s2_xy": 0.5 * (
                np.mean([c["s2_x"] for c in gen_s2_list], axis=0)
                + np.mean([c["s2_y"] for c in gen_s2_list], axis=0)
            ),
        }
        s2_store[state_name] = s2_entry

        s2_rmse_z = float(np.sqrt(np.mean((s2_entry["real_s2_z"] - s2_entry["gen_s2_z"]) ** 2)))
        s2_rmse_xy = float(np.sqrt(np.mean((s2_entry["real_s2_xy"] - s2_entry["gen_s2_xy"]) ** 2)))

        full_report[state_name] = {
            "load_mv_per_v": load_val,
            "is_heldout_interpolation": state_name in heldout,
            "num_real": int(real_vols.shape[0]),
            "num_generated": int(gen_vols.shape[0]),
            "s2_rmse_z": s2_rmse_z,
            "s2_rmse_xy": s2_rmse_xy,
            "real": real_sum,
            "generated": gen_sum,
        }

        row: dict[str, object] = {
            "state": state_name,
            "load_mv_per_v": load_val,
            "regime": "heldout_interp" if state_name in heldout else "train_load",
            "real_solid_mean": round(real_sum["solid_fraction"]["mean"], 5),
            "real_solid_std": round(real_sum["solid_fraction"]["std"], 5),
            "gen_solid_mean": round(gen_sum["solid_fraction"]["mean"], 5),
            "gen_solid_std": round(gen_sum["solid_fraction"]["std"], 5),
            "real_ssa_mean": round(real_sum["ssa_total"]["mean"], 5),
            "gen_ssa_mean": round(gen_sum["ssa_total"]["mean"], 5),
            "real_pore_anisotropy": round(real_sum["pore_anisotropy_xy_over_z"]["mean"], 4),
            "gen_pore_anisotropy": round(gen_sum["pore_anisotropy_xy_over_z"]["mean"], 4),
            "real_tortuosity_z": round(real_sum["geodesic_tortuosity_z"]["mean"], 4),
            "gen_tortuosity_z": round(gen_sum["geodesic_tortuosity_z"]["mean"], 4),
            "real_solid_conn": round(real_sum["solid_connected_fraction"]["mean"], 4),
            "gen_solid_conn": round(gen_sum["solid_connected_fraction"]["mean"], 4),
            "s2_rmse_z": round(s2_rmse_z, 6),
            "s2_rmse_xy": round(s2_rmse_xy, 6),
        }
        comparison_rows.append(row)
        print(json.dumps(row), flush=True)

    with (eval_dir / "physics_validation_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)

    with (eval_dir / "physics_validation_report.json").open("w") as f:
        json.dump(full_report, f, indent=2)

    # Plot 1: Multi-metric compression response curves (Real vs Generated)
    loads_x = [float(r["load_mv_per_v"]) for r in comparison_rows]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    panels = [
        (axes[0, 0], "solid_fraction", "Solid Volume Fraction φ_s"),
        (axes[0, 1], "ssa_total", f"Specific Surface Area (1/{args.voxel_size_um:g} µm)"),
        (axes[1, 0], "pore_anisotropy_xy_over_z", "Pore Chord Anisotropy (XY / Z)"),
        (axes[1, 1], "geodesic_tortuosity_z", "Through-Plane Geodesic Tortuosity τ_z"),
    ]
    ordered_names = [str(r["state"]) for r in comparison_rows]
    for ax, metric_key, ylabel in panels:
        r_m = [full_report[s]["real"][metric_key]["mean"] for s in ordered_names]  # type: ignore[index]
        r_s = [full_report[s]["real"][metric_key]["std"] for s in ordered_names]  # type: ignore[index]
        g_m = [full_report[s]["generated"][metric_key]["mean"] for s in ordered_names]  # type: ignore[index]
        g_s = [full_report[s]["generated"][metric_key]["std"] for s in ordered_names]  # type: ignore[index]

        ax.errorbar(loads_x, r_m, yerr=r_s, fmt="-o", capsize=4, lw=1.8, label="Real XCT")
        ax.errorbar(loads_x, g_m, yerr=g_s, fmt="--s", capsize=4, lw=1.8, label="PorousDiff Generated")
        for x_val, s_name in zip(loads_x, ordered_names):
            if s_name in heldout:
                ax.axvline(x_val, color="tab:red", ls=":", alpha=0.5)
        ax.set_xlabel("Load-Cell Compression Reading (mV/V)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("PorousDiff Physics Validation Across Mechanical Compression", fontsize=13)
    fig.tight_layout()
    fig.savefig(eval_dir / "physics_compression_curves.png", dpi=240)
    fig.savefig(eval_dir / "physics_compression_curves.pdf")
    plt.close(fig)

    # Plot 2: Directional S_2(r) comparison across states
    n_st = len(ordered_names)
    fig, axes = plt.subplots(1, n_st, figsize=(4.5 * n_st, 4.2), sharey=False)
    if n_st == 1:
        axes = [axes]
    for ax, s_name in zip(axes, ordered_names):
        ent = s2_store[s_name]
        r = ent["lags"] * args.voxel_size_um
        ax.plot(r, ent["real_s2_xy"], "-", color="tab:blue", lw=1.8, label="Real S₂(XY)")
        ax.plot(r, ent["gen_s2_xy"], "--", color="tab:blue", lw=1.8, label="Gen S₂(XY)")
        ax.plot(r, ent["real_s2_z"], "-", color="tab:orange", lw=1.8, label="Real S₂(Z)")
        ax.plot(r, ent["gen_s2_z"], "--", color="tab:orange", lw=1.8, label="Gen S₂(Z)")
        tag = " [Held-Out]" if s_name in heldout else ""
        ax.set_title(f"{s_name} ({states[s_name]:g} mV/V){tag}")
        ax.set_xlabel(f"Lag r ({args.voxel_size_um:g} µm)" if args.voxel_size_um != 1.0 else "Lag r (voxels)")
        ax.set_ylabel("Two-Point Probability S₂(r)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(eval_dir / "s2_correlation_comparison.png", dpi=240)
    fig.savefig(eval_dir / "s2_correlation_comparison.pdf")
    plt.close(fig)
    print(f"Saved validation reports & figures -> {eval_dir}", flush=True)


if __name__ == "__main__":
    main()
