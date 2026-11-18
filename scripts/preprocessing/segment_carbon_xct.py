"""Calibrated binary XCT segmentation with circular ROI masking and Z-artifact gating.

Preserves physical solid-fraction scaling across compression states by subtracting
each slice's in-cylinder pore background mode and applying a single shared threshold
across all loads, while flagging bullseye ring seams and rig plungers in `valid_slices.json`.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from porousdiff.data import default_project_root
from porousdiff.preprocessing import (
    calibrate_delta_threshold,
    compute_radial_ring_score,
    detect_valid_z_intervals,
    make_circular_mask,
    resolve_tiff_directory,
    segment_slice_calibrated,
)


def _validate_token(value: str, label: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError(f"Invalid {label}: {value!r}")


def _save_z_diagnostic_plot(
    out_path: Path,
    dataset_name: str,
    solid_fractions: list[float],
    ring_scores: list[float],
    slice_jumps: list[float],
    valid_mask: np.ndarray,
    intervals: list[tuple[int, int]],
) -> None:
    z = np.arange(len(solid_fractions))
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    axes[0].plot(z, solid_fractions, color="tab:blue", lw=1.3, label="In-cylinder solid fraction")
    for idx, (s, e) in enumerate(intervals):
        axes[0].axvspan(
            s, e - 1, color="tab:green", alpha=0.15, label="Accepted clean Z" if idx == 0 else None
        )
    axes[0].set_ylabel("Solid fraction")
    axes[0].set_title(f"{dataset_name} — Segmented Z Profile & Accepted Clean Intervals")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(z, ring_scores, color="tab:red", lw=1.2, label="Radial ring score")
    ax2_r = ax2.twinx()
    ax2_r.plot(z, slice_jumps, color="tab:orange", lw=1.0, alpha=0.75, label="|dI/dz| slice jump")
    bad_idx = np.where(~valid_mask)[0]
    if bad_idx.size:
        ax2.scatter(
            bad_idx,
            np.asarray(ring_scores)[bad_idx],
            color="black",
            s=6,
            zorder=4,
            label="Excluded artifact/plunger slices",
        )
    ax2.set_xlabel("Slice index Z")
    ax2.set_ylabel("Radial ring score")
    ax2_r.set_ylabel("Mean |dI/dz|")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def process_single_dataset(
    dataset_name: str,
    data_root: Path,
    output_root: Path,
    version: str,
    delta_threshold: float,
    sigma: float,
    roi_radius_fraction: float,
    expected_slices: int,
    min_interval_length: int,
    max_ring_score: float,
    seam_margin: int,
    overwrite: bool,
) -> dict[str, object]:
    input_base = data_root / dataset_name
    tiff_dir = resolve_tiff_directory(input_base)
    files = sorted(list(tiff_dir.glob("*.tif")) + list(tiff_dir.glob("*.tiff")))

    if expected_slices > 0 and len(files) != expected_slices:
        raise RuntimeError(
            f"[{dataset_name}] Expected {expected_slices} TIFF slices in {tiff_dir}, found {len(files)}"
        )
    if not files:
        raise RuntimeError(f"[{dataset_name}] No TIFF slices found in {tiff_dir}")

    output_dir = output_root / f"{dataset_name}_binary_corrected_{version}"
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_dir}. Pass --overwrite or a new --version."
        )

    with tifffile.TiffFile(files[0]) as first_tif:
        shape_hw = tuple(first_tif.pages[0].shape)
        dtype = first_tif.pages[0].dtype
    if len(shape_hw) != 2:
        raise ValueError(f"[{dataset_name}] Expected 2D TIFF slices, got shape {shape_hw}")

    for path in files:
        with tifffile.TiffFile(path) as tif:
            page = tif.pages[0]
            if tuple(page.shape) != shape_hw or page.dtype != dtype:
                raise ValueError(f"[{dataset_name}] Inconsistent TIFF format at {path.name}")

    output_dir.mkdir(parents=True, exist_ok=True)
    roi_mask = make_circular_mask(shape_hw, radius_fraction=roi_radius_fraction)

    solid_fractions: list[float] = []
    pore_modes: list[float] = []
    ring_scores: list[float] = []
    slice_jumps: list[float] = []
    prev_sub: np.ndarray | None = None

    for idx, path in enumerate(files, start=1):
        image = tifffile.imread(path)
        binary, stats = segment_slice_calibrated(
            image,
            delta_threshold=delta_threshold,
            roi_mask=roi_mask,
            sigma=sigma,
        )
        tifffile.imwrite(output_dir / f"binary_{idx:04d}.tif", binary)

        solid_fractions.append(stats["solid_fraction"])
        pore_modes.append(stats["pore_mode"])
        ring_scores.append(compute_radial_ring_score(image))

        curr_sub = image[::4, ::4].astype(np.float32)
        if prev_sub is None:
            slice_jumps.append(0.0)
        else:
            slice_jumps.append(float(np.abs(curr_sub - prev_sub).mean()))
        prev_sub = curr_sub

        if idx % 50 == 0 or idx == len(files):
            print(f"[{dataset_name}] Processed {idx}/{len(files)} slices", flush=True)

    if len(slice_jumps) > 1:
        slice_jumps[0] = slice_jumps[1]

    valid_mask, valid_intervals = detect_valid_z_intervals(
        ring_scores=ring_scores,
        pore_modes=pore_modes,
        slice_jumps=slice_jumps,
        min_interval_length=min_interval_length,
        max_ring_score=max_ring_score,
        seam_margin=seam_margin,
    )

    sf_arr = np.asarray(solid_fractions, dtype=np.float64)
    clean_sf = sf_arr[valid_mask] if valid_mask.any() else sf_arr

    valid_slices_payload = {
        "dataset": dataset_name,
        "version": version,
        "total_slices": len(files),
        "slice_shape_hw": list(shape_hw),
        "roi_radius_fraction": roi_radius_fraction,
        "min_interval_length": min_interval_length,
        "seam_margin": seam_margin,
        "max_ring_score": max_ring_score,
        "valid_slice_count": int(valid_mask.sum()),
        "valid_z_intervals_0based_half_open": [[int(s), int(e)] for s, e in valid_intervals],
    }
    with (output_dir / "valid_slices.json").open("w") as vf:
        json.dump(valid_slices_payload, vf, indent=2)

    _save_z_diagnostic_plot(
        out_path=output_dir / "z_quality_diagnostic.png",
        dataset_name=dataset_name,
        solid_fractions=solid_fractions,
        ring_scores=ring_scores,
        slice_jumps=slice_jumps,
        valid_mask=valid_mask,
        intervals=valid_intervals,
    )

    report: dict[str, object] = {
        "dataset": dataset_name,
        "version": version,
        "input_directory": str(tiff_dir),
        "output_directory": str(output_dir),
        "slices_total": len(files),
        "slices_valid_clean": int(valid_mask.sum()),
        "valid_z_intervals": [[int(s), int(e)] for s, e in valid_intervals],
        "slice_shape_hw": list(shape_hw),
        "input_dtype": str(dtype),
        "sigma": sigma,
        "delta_threshold": delta_threshold,
        "roi_radius_fraction": roi_radius_fraction,
        "threshold_method": "Pore-mode-referenced shared physical delta threshold",
        "clean_solid_fraction_mean": float(clean_sf.mean()),
        "clean_solid_fraction_std": float(clean_sf.std()),
        "clean_solid_fraction_min": float(clean_sf.min()),
        "clean_solid_fraction_max": float(clean_sf.max()),
        "all_slices_solid_fraction_mean": float(sf_arr.mean()),
    }
    with (output_dir / "segmentation_report.json").open("w") as rf:
        json.dump(report, rf, indent=2)

    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    default_root = default_project_root() / "data" / "carbon_virgin"
    parser = argparse.ArgumentParser(
        description="Physics-consistent XCT segmentation with circular ROI and Z-artifact gating"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", help="Single dataset folder name (e.g. Op008mVV)")
    group.add_argument("--datasets", nargs="+", help="Multiple dataset folders to segment consistently")

    parser.add_argument("--data-root", type=Path, default=default_root)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--version", default="carbon_calibrated_v1")
    parser.add_argument(
        "--delta-threshold",
        type=float,
        default=5500.0,
        help="Shared intensity threshold above local pore mode (used across all loads)",
    )
    parser.add_argument(
        "--auto-calibrate-from",
        default=None,
        help="Optional dataset name from which to auto-calibrate delta_threshold before segmenting",
    )
    parser.add_argument("--sigma", type=float, default=32.0)
    parser.add_argument("--roi-radius-fraction", type=float, default=0.70)
    parser.add_argument(
        "--expected-slices",
        type=int,
        default=0,
        help="Expected slice count (0 = accept any complete sequence found)",
    )
    parser.add_argument("--min-interval-length", type=int, default=96)
    parser.add_argument("--max-ring-score", type=float, default=0.15)
    parser.add_argument("--seam-margin", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else list(args.datasets)
    for ds in datasets:
        _validate_token(ds, "dataset")
    _validate_token(args.version, "version")

    if args.sigma < 0 or args.expected_slices < 0 or args.min_interval_length < 1:
        parser.error("Invalid non-positive numeric parameter")

    out_root = args.output_root if args.output_root is not None else args.data_root
    threshold = args.delta_threshold

    if args.auto_calibrate_from:
        _validate_token(args.auto_calibrate_from, "auto-calibrate-from")
        ref_dir = resolve_tiff_directory(args.data_root / args.auto_calibrate_from)
        ref_files = sorted(list(ref_dir.glob("*.tif")) + list(ref_dir.glob("*.tiff")))
        if not ref_files:
            raise RuntimeError(f"No reference TIFFs found in {ref_dir}")
        step = max(1, len(ref_files) // 16)
        sample_imgs = [
            tifffile.imread(p)
            for p in ref_files[::step]
            if compute_radial_ring_score(tifffile.imread(p)) <= args.max_ring_score
        ]
        if not sample_imgs:
            sample_imgs = [tifffile.imread(ref_files[len(ref_files) // 4])]
        roi = make_circular_mask(sample_imgs[0].shape, radius_fraction=args.roi_radius_fraction)
        threshold = calibrate_delta_threshold(sample_imgs, roi_mask=roi, sigma=args.sigma)
        print(f"Auto-calibrated shared delta_threshold = {threshold:.2f}", flush=True)

    for ds in datasets:
        process_single_dataset(
            dataset_name=ds,
            data_root=args.data_root,
            output_root=out_root,
            version=args.version,
            delta_threshold=threshold,
            sigma=args.sigma,
            roi_radius_fraction=args.roi_radius_fraction,
            expected_slices=args.expected_slices,
            min_interval_length=args.min_interval_length,
            max_ring_score=args.max_ring_score,
            seam_margin=args.seam_margin,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
