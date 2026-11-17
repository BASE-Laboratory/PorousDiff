import argparse
import json
import re
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu

DATA_ROOT = Path("/projects/u6uc/asif/data/carbon_virgin")


def segment_binary(image, sigma=30):
    image = image.astype(np.float32)

    background = gaussian_filter(image, sigma=sigma)
    corrected = image - background

    lo, hi = np.percentile(corrected, [1, 99])

    norm = np.clip(
        (corrected - lo) / (hi - lo + 1e-8),
        0, 1
    )

    threshold = threshold_otsu(norm)
    return (norm > threshold).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(
        description="Reusable binary XCT segmentation"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--expected-slices", type=int, required=True)
    parser.add_argument("--sigma", type=float, default=50)
    parser.add_argument("--version", default="v2")
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.dataset):
        parser.error("Invalid dataset name")

    if args.expected_slices <= 0 or args.sigma <= 0:
        parser.error("Expected slices and sigma must be positive")

    input_dir = DATA_ROOT / args.dataset
    output_dir = DATA_ROOT / (
        f"{args.dataset}_binary_corrected_{args.version}"
    )

    files = sorted(input_dir.glob("*.tif"))

    if len(files) != args.expected_slices:
        raise RuntimeError(
            f"Expected {args.expected_slices} TIFFs, "
            f"found {len(files)}"
        )

    if output_dir.exists():
        raise FileExistsError(
            f"Output already exists: {output_dir}"
        )

    # Validate input metadata before creating output.
    for path in files:
        with tifffile.TiffFile(path) as tif:
            page = tif.pages[0]
            if page.shape != (2560, 2560) or page.dtype != np.uint16:
                raise ValueError(f"Unexpected TIFF format: {path.name}")

    output_dir.mkdir(parents=True)

    fractions = []

    for i, path in enumerate(files, start=1):
        image = tifffile.imread(path)
        binary = segment_binary(image, args.sigma)

        destination = output_dir / f"binary_{i:04d}.tif"
        tifffile.imwrite(destination, binary)

        fractions.append(float(binary.mean()))

        if i % 25 == 0:
            print(
                f"Processed {i}/{len(files)} slices",
                flush=True
            )

    report = {
        "dataset": args.dataset,
        "input_directory": str(input_dir),
        "output_directory": str(output_dir),
        "slices": len(files),
        "sigma": args.sigma,
        "normalization_percentiles": [1, 99],
        "threshold": "Otsu per slice",
        "white_fraction_mean": float(np.mean(fractions)),
        "white_fraction_min": float(np.min(fractions)),
        "white_fraction_max": float(np.max(fractions)),
        "white_fraction_std": float(np.std(fractions)),
    }

    with open(output_dir / "segmentation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
