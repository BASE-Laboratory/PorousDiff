"""Extract stratified 3D binary XCT patches inside the circular ROI across clean Z intervals.

Fixes Z-depth starvation, exterior-air corner contamination, and adjacent train/val Z leakage,
and writes stacked `patches.npy` (`uint8`) for fast single-read RAM loading on HPC.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import tifffile

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from porousdiff.data import default_project_root
from porousdiff.preprocessing import (
    build_stratified_patch_plan,
    enumerate_cylinder_xy_positions,
    split_z_intervals_with_buffer,
)


def _validate_token(value: str, label: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError(f"Invalid {label}: {value!r}")


def extract_dataset_patches(
    dataset_name: str,
    data_root: Path,
    output_root: Path,
    version: str,
    expected_slices: int,
    patch_size: int,
    stride_xy: int,
    stride_z: int,
    max_patches: int,
    min_white: float,
    max_white: float,
    val_fraction: float,
    roi_radius_fraction: float,
    ignore_valid_slices: bool,
    save_individual_npy: bool,
    seed: int,
    overwrite: bool,
) -> dict[str, object]:
    input_dir = data_root / f"{dataset_name}_binary_corrected_{version}"
    output_dir = output_root / f"{dataset_name}_binary_patches{patch_size}_{version}"

    files = sorted(input_dir.glob("binary_*.tif"))
    if not files:
        raise FileNotFoundError(f"No binary_*.tif files found in {input_dir}")

    if expected_slices > 0 and len(files) != expected_slices:
        raise RuntimeError(f"Expected {expected_slices} slices in {input_dir}, found {len(files)}")

    expected_names = [f"binary_{i:04d}.tif" for i in range(1, len(files) + 1)]
    if [f.name for f in files] != expected_names:
        raise RuntimeError(f"Binary TIFF sequence in {input_dir} is incomplete or non-contiguous")

    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_dir}. Use a new --version or --overwrite."
        )

    first_slice = tifffile.imread(files[0])
    shape_hw = (int(first_slice.shape[0]), int(first_slice.shape[1]))

    valid_slices_file = input_dir / "valid_slices.json"
    if valid_slices_file.is_file() and not ignore_valid_slices:
        payload = json.loads(valid_slices_file.read_text())
        clean_intervals = [
            (int(s), int(e)) for s, e in payload["valid_z_intervals_0based_half_open"]
        ]
        roi_radius_fraction = float(payload.get("roi_radius_fraction", roi_radius_fraction))
    else:
        clean_intervals = [(0, len(files))]

    split_intervals = split_z_intervals_with_buffer(
        intervals=clean_intervals,
        patch_size=patch_size,
        val_fraction=val_fraction,
        buffer_slices=patch_size,
    )

    xy_positions = enumerate_cylinder_xy_positions(
        shape_hw=shape_hw,
        patch_size=patch_size,
        stride_xy=stride_xy,
        radius_fraction=roi_radius_fraction,
    )
    if not xy_positions:
        raise RuntimeError(
            f"No {patch_size}x{patch_size} XY patches fit inside circular ROI "
            f"(slice shape={shape_hw}, radius_fraction={roi_radius_fraction})"
        )

    budgets = {"train": int(round(max_patches * (1.0 - val_fraction)))}
    budgets["val"] = max(1, max_patches - budgets["train"])

    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "patch_metadata.csv"

    counts = {"train": 0, "val": 0}
    solid_stats: dict[str, list[float]] = {"train": [], "val": []}
    sampled_z_windows: dict[str, set[int]] = {"train": set(), "val": set()}

    with metadata_path.open("w", newline="") as mf:
        writer = csv.writer(mf)
        writer.writerow(
            ["filename", "split", "z_start", "z_end", "y_start", "x_start", "white_fraction"]
        )

        for split in ("train", "val"):
            split_dir = output_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)

            plan = build_stratified_patch_plan(
                z_intervals=split_intervals[split],
                xy_positions=xy_positions,
                patch_size=patch_size,
                stride_z=stride_z,
                max_patches=budgets[split],
                rng=rng,
            )

            collected: list[np.ndarray] = []
            # Because `plan` is sorted by ascending `z`, maintain a sliding slab cache
            cached_z: int | None = None
            cached_slab: np.ndarray | None = None

            for z, xy_list in plan:
                if counts[split] >= budgets[split]:
                    break

                if cached_slab is not None and cached_z is not None and z > cached_z and (z - cached_z) < patch_size:
                    overlap = patch_size - (z - cached_z)
                    new_slices = [
                        tifffile.imread(f)
                        for f in files[cached_z + patch_size : z + patch_size]
                    ]
                    slab = np.concatenate([cached_slab[-overlap:], np.stack(new_slices, axis=0)], axis=0)
                else:
                    slab = np.stack([tifffile.imread(f) for f in files[z : z + patch_size]], axis=0)

                cached_z = z
                cached_slab = slab

                if slab.dtype != np.uint8 or slab.shape != (patch_size, shape_hw[0], shape_hw[1]):
                    raise ValueError(f"Unexpected slab shape/dtype at z={z}: {slab.shape} {slab.dtype}")
                if not np.all((slab == 0) | (slab == 1)):
                    raise ValueError(f"Non-binary values encountered at z={z}")

                for y, x in xy_list:
                    patch = slab[:, y : y + patch_size, x : x + patch_size]
                    wf = float(patch.mean())
                    if not (min_white <= wf <= max_white):
                        continue

                    idx = counts[split]
                    filename = f"{dataset_name}_binary_{split}_{idx:05d}.npy"
                    if save_individual_npy:
                        np.save(split_dir / filename, patch)

                    collected.append(patch.copy())
                    solid_stats[split].append(wf)
                    sampled_z_windows[split].add(z)
                    writer.writerow([filename, split, z + 1, z + patch_size, y, x, round(wf, 6)])
                    counts[split] += 1

                    if counts[split] >= budgets[split]:
                        break

            if not collected:
                raise RuntimeError(f"[{dataset_name}] Extracted 0 patches for split={split!r}")

            stacked = np.stack(collected, axis=0).astype(np.uint8)
            np.save(split_dir / "patches.npy", stacked)
            print(
                f"[{dataset_name}] {split}: {counts[split]}/{budgets[split]} patches across "
                f"{len(sampled_z_windows[split])} Z-windows (mean solid={np.mean(solid_stats[split]):.4f})",
                flush=True,
            )

    report: dict[str, object] = {
        "dataset": dataset_name,
        "version": version,
        "input_directory": str(input_dir),
        "output_directory": str(output_dir),
        "patch_size": patch_size,
        "stride_xy": stride_xy,
        "stride_z": stride_z,
        "roi_radius_fraction": roi_radius_fraction,
        "xy_candidates_per_z": len(xy_positions),
        "clean_z_intervals": [[s, e] for s, e in clean_intervals],
        "split_z_intervals_0based_half_open": {
            k: [[s, e] for s, e in v] for k, v in split_intervals.items()
        },
        "z_windows_sampled": {k: sorted(list(v)) for k, v in sampled_z_windows.items()},
        "min_white": min_white,
        "max_white": max_white,
        "val_fraction": val_fraction,
        "seed": seed,
        "counts": counts,
        "total_patches": sum(counts.values()),
        "mean_solid_fraction": {
            k: float(np.mean(v)) if v else 0.0 for k, v in solid_stats.items()
        },
        "std_solid_fraction": {
            k: float(np.std(v)) if v else 0.0 for k, v in solid_stats.items()
        },
    }
    with (output_dir / "patch_extraction_report.json").open("w") as rf:
        json.dump(report, rf, indent=2)

    return report


def main() -> None:
    default_root = default_project_root() / "data" / "carbon_virgin"
    parser = argparse.ArgumentParser(
        description="Extract Z-stratified 3D binary XCT patches inside circular sample ROI"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", help="Single dataset name (e.g. Op008mVV)")
    group.add_argument("--datasets", nargs="+", help="Multiple dataset names to extract")

    parser.add_argument("--data-root", type=Path, default=default_root)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--version", default="carbon_calibrated_v1")
    parser.add_argument("--expected-slices", type=int, default=0)
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--stride-xy", type=int, default=96)
    parser.add_argument("--stride-z", type=int, default=32)
    parser.add_argument("--max-patches", type=int, default=1200)
    parser.add_argument("--min-white", type=float, default=0.002)
    parser.add_argument("--max-white", type=float, default=0.95)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--roi-radius-fraction", type=float, default=0.70)
    parser.add_argument("--ignore-valid-slices", action="store_true")
    parser.add_argument(
        "--no-individual-npy",
        action="store_true",
        help="Write only stacked patches.npy (recommended for large HPC runs)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else list(args.datasets)
    for ds in datasets:
        _validate_token(ds, "dataset")
    _validate_token(args.version, "version")

    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be between 0 and 1")
    if not 0.0 <= args.min_white < args.max_white <= 1.0:
        raise ValueError("Invalid white-fraction thresholds")
    if min(args.patch_size, args.stride_xy, args.stride_z, args.max_patches) <= 0:
        raise ValueError("Patch size, strides, and max-patches must be positive")

    out_root = args.output_root if args.output_root is not None else args.data_root
    for ds in datasets:
        extract_dataset_patches(
            dataset_name=ds,
            data_root=args.data_root,
            output_root=out_root,
            version=args.version,
            expected_slices=args.expected_slices,
            patch_size=args.patch_size,
            stride_xy=args.stride_xy,
            stride_z=args.stride_z,
            max_patches=args.max_patches,
            min_white=args.min_white,
            max_white=args.max_white,
            val_fraction=args.val_fraction,
            roi_radius_fraction=args.roi_radius_fraction,
            ignore_valid_slices=args.ignore_valid_slices,
            save_individual_npy=not args.no_individual_npy,
            seed=args.seed,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
