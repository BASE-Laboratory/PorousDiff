
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tifffile


def main():
    parser = argparse.ArgumentParser(
        description="Extract reusable 3D binary XCT patches"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--version", default="v2")
    parser.add_argument("--expected-slices", type=int, default=800)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--stride-xy", type=int, default=128)
    parser.add_argument("--stride-z", type=int, default=32)
    parser.add_argument("--max-patches", type=int, default=1200)
    parser.add_argument("--min-white", type=float, default=0.01)
    parser.add_argument("--max-white", type=float, default=0.99)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # --------------------------------------------------
    # 1. Validate arguments
    # --------------------------------------------------
    if not args.dataset.isalnum():
        raise ValueError("Dataset must contain only letters and numbers")

    if not args.version.replace("_", "").isalnum():
        raise ValueError("Version must contain only letters and numbers")

    if not 0 < args.val_fraction < 1:
        raise ValueError("val-fraction must be between 0 and 1")

    if not 0 <= args.min_white < args.max_white <= 1:
        raise ValueError("Invalid white-fraction thresholds")

    if min(
        args.expected_slices,
        args.patch_size,
        args.stride_xy,
        args.stride_z,
        args.max_patches,
    ) <= 0:
        raise ValueError("Sizes, strides and patch count must be positive")

    # --------------------------------------------------
    # 2. Input and output paths
    # --------------------------------------------------
    root = Path("/projects/u6uc/asif/data/carbon_virgin")

    input_dir = root / (
        f"{args.dataset}_binary_corrected_{args.version}"
    )

    output_dir = root / (
        f"{args.dataset}_binary_patches"
        f"{args.patch_size}_{args.version}"
    )

    files = sorted(input_dir.glob("binary_*.tif"))

    if len(files) != args.expected_slices:
        raise RuntimeError(
            f"Expected {args.expected_slices} slices, "
            f"found {len(files)}"
        )

    expected_names = [
        f"binary_{i:04d}.tif"
        for i in range(1, args.expected_slices + 1)
    ]

    if [f.name for f in files] != expected_names:
        raise RuntimeError("Binary TIFF sequence is incomplete")

    # Never overwrite an existing experiment.
    if output_dir.exists():
        raise FileExistsError(
            f"Output already exists: {output_dir}. "
            "Use a new version for a separate experiment."
        )

    # --------------------------------------------------
    # 3. Spatial train/validation split
    # --------------------------------------------------
    size = args.patch_size
    total = len(files)

    boundary = int(total * (1 - args.val_fraction))

    regions = {
        "train": (0, boundary),
        "val": (boundary, total),
    }

    budgets = {
        "train": int(
            args.max_patches * (1 - args.val_fraction)
        )
    }

    budgets["val"] = (
        args.max_patches - budgets["train"]
    )

    for split, (start, end) in regions.items():
        if end - start < size:
            raise RuntimeError(
                f"{split} region has fewer than "
                f"{size} slices"
            )

    # --------------------------------------------------
    # 4. Initialize reproducible sampling
    # --------------------------------------------------
    rng = np.random.default_rng(args.seed)

    counts = {
        "train": 0,
        "val": 0,
    }

    output_dir.mkdir(parents=True)

    metadata_path = output_dir / "patch_metadata.csv"

    # --------------------------------------------------
    # 5. Extract patches
    # --------------------------------------------------
    with metadata_path.open("w", newline="") as metadata_file:
        writer = csv.writer(metadata_file)

        writer.writerow([
            "filename",
            "split",
            "z_start",
            "z_end",
            "y_start",
            "x_start",
            "white_fraction",
        ])

        for split, (start, end) in regions.items():
            split_dir = output_dir / split
            split_dir.mkdir()

            # Shuffle depth windows.
            z_positions = list(
                range(
                    start,
                    end - size + 1,
                    args.stride_z,
                )
            )

            rng.shuffle(z_positions)

            print(
                f"Starting {split}: "
                f"{len(z_positions)} depth windows",
                flush=True,
            )

            for z in z_positions:
                # Read only 64 consecutive TIFF slices.
                volume = np.stack([
                    tifffile.imread(f)
                    for f in files[z:z + size]
                ])

                if volume.dtype != np.uint8:
                    raise ValueError(
                        f"Unexpected data type at z={z}"
                    )

                if volume.ndim != 3:
                    raise ValueError(
                        f"Expected 3D volume at z={z}"
                    )

                if volume.shape[0] != size:
                    raise ValueError(
                        f"Unexpected depth at z={z}"
                    )

                if not np.all(
                    (volume == 0) | (volume == 1)
                ):
                    raise ValueError(
                        f"Non-binary values at z={z}"
                    )

                # Build XY sampling positions.
                xy_positions = [
                    (y, x)
                    for y in range(
                        0,
                        volume.shape[1] - size + 1,
                        args.stride_xy,
                    )
                    for x in range(
                        0,
                        volume.shape[2] - size + 1,
                        args.stride_xy,
                    )
                ]

                # Shuffle XY positions as well.
                rng.shuffle(xy_positions)

                for y, x in xy_positions:
                    patch = volume[
                        :,
                        y:y + size,
                        x:x + size,
                    ]

                    white_fraction = float(patch.mean())

                    # Reject nearly empty or nearly full patches.
                    if not (
                        args.min_white
                        < white_fraction
                        < args.max_white
                    ):
                        continue

                    filename = (
                        f"{args.dataset}_binary_{split}_"
                        f"{counts[split]:05d}.npy"
                    )

                    np.save(
                        split_dir / filename,
                        patch,
                    )

                    writer.writerow([
                        filename,
                        split,
                        z + 1,
                        z + size,
                        y,
                        x,
                        round(white_fraction, 6),
                    ])

                    counts[split] += 1

                    if counts[split] >= budgets[split]:
                        break

                # Stop once this split's budget is reached.
                if counts[split] >= budgets[split]:
                    break

            print(
                f"{split}: "
                f"{counts[split]}/{budgets[split]} patches",
                flush=True,
            )

    # --------------------------------------------------
    # 6. Save experiment report
    # --------------------------------------------------
    report = {
        "dataset": args.dataset,
        "version": args.version,
        "input_directory": str(input_dir),
        "output_directory": str(output_dir),
        "expected_slices": args.expected_slices,
        "patch_size": size,
        "stride_xy": args.stride_xy,
        "stride_z": args.stride_z,
        "max_patches": args.max_patches,
        "min_white": args.min_white,
        "max_white": args.max_white,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "train_z_range": [1, boundary],
        "val_z_range": [boundary + 1, total],
        "counts": counts,
        "total_patches": sum(counts.values()),
    }

    with (
        output_dir / "patch_extraction_report.json"
    ).open("w") as report_file:
        json.dump(report, report_file, indent=2)

    print("\nPatch extraction completed.")
    print("Training patches:", counts["train"])
    print("Validation patches:", counts["val"])
    print("Total patches:", sum(counts.values()))
    print("Output:", output_dir)


if __name__ == "__main__":
    main()
