"""Stream-segment raw XCT scans on a local drive (e.g. D:\\dry_samples) directly into uploadable 3D patch bundles.

Avoids transferring 300+ GB of raw 16-bit Paganin TIFFs or writing hundreds of GBs of intermediate
segmented slices to disk:
  1. Auto-discovers `crbon_virgin_*` (or specified) scan folders containing `.tif` / `rec_16bit_Paganin/*.tif`.
  2. Performs a single pass over slices to calibrate Z-artifact & plunger gating (`z_quality_diagnostic.png`,
     `segmentation_report.json`, `valid_slices.json`).
  3. Streams clean Z-slabs directly into Z-stratified `train/patches.npy` and `val/patches.npy` (`~80 MB` compressed per load).
  4. Writes `primary_states.json` + `bundle_manifest.json` (separating primary `crbon_virgin_1` loads from `rep2` / `crbon_virgin_2` repeats)
     and an optional single `<output_dir>.zip` ready for `scp`/`rsync` to Isambard-AI.

Spyder usage:
    Open this file in Spyder, check `SPYDER_DRIVE_ROOT` / `SPYDER_OUTPUT_DIR` below, and press F5 (Run).
CLI usage:
    python scripts/preprocessing/prepare_local_drive_bundle.py --drive-root "D:/dry_samples" --output-dir "D:/porousdiff_bundle" --zip
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from porousdiff.preprocessing import (
    build_stratified_patch_plan,
    compute_radial_ring_score,
    detect_valid_z_intervals,
    enumerate_cylinder_xy_positions,
    make_circular_mask,
    parse_load_value,
    resolve_tiff_directory,
    segment_slice_calibrated,
    split_z_intervals_with_buffer,
)

# Defaults used only when invoked with zero CLI flags inside Spyder (`runfile(...)`)
SPYDER_DRIVE_ROOT = Path("D:/dry_samples")
SPYDER_OUTPUT_DIR = Path("D:/porousdiff_upload_bundle")


def _clean_state_name(folder_name: str) -> str:
    return folder_name.strip().rstrip("_")


def _is_repeat_or_secondary(folder_name: str) -> bool:
    low = folder_name.lower()
    return ("rep" in low) or ("virgin_2" in low) or ("sample2" in low)


def _save_z_plot(
    out_path: Path,
    state_name: str,
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
    axes[0].set_title(f"{state_name} — Segmented Z Profile & Accepted Clean Intervals")
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


def discover_scan_folders(drive_root: Path) -> list[Path]:
    candidates: list[Path] = []
    for entry in sorted(drive_root.iterdir()):
        if not entry.is_dir():
            continue
        try:
            resolve_tiff_directory(entry)
            parse_load_value(entry.name)
            candidates.append(entry)
        except (FileNotFoundError, ValueError):
            continue
    return candidates


def stream_bundle_single_scan(
    scan_dir: Path,
    output_dir: Path,
    version: str = "carbon_calibrated_v1",
    delta_threshold: float = 5500.0,
    sigma: float = 32.0,
    roi_radius_fraction: float = 0.70,
    patch_size: int = 96,
    stride_xy: int = 96,
    stride_z: int = 32,
    max_patches: int = 1200,
    min_white: float = 0.005,
    max_white: float = 0.90,
    val_fraction: float = 0.2,
    min_interval_length: int = 96,
    max_ring_score: float = 0.15,
    seam_margin: int = 12,
    seed: int = 42,
) -> dict[str, object]:
    state_name = _clean_state_name(scan_dir.name)
    load_mv = parse_load_value(state_name)
    tiff_dir = resolve_tiff_directory(scan_dir)
    files = sorted(list(tiff_dir.glob("*.tif")) + list(tiff_dir.glob("*.tiff")))
    if len(files) < 2 * patch_size:
        raise RuntimeError(
            f"[{state_name}] Need >= {2 * patch_size} slices in {tiff_dir}, found {len(files)}"
        )

    with tifffile.TiffFile(files[0]) as first_tif:
        shape_hw = tuple(first_tif.pages[0].shape)
        dtype = first_tif.pages[0].dtype

    roi_mask = make_circular_mask(shape_hw, radius_fraction=roi_radius_fraction)
    xy_positions = enumerate_cylinder_xy_positions(
        shape_hw=shape_hw,
        patch_size=patch_size,
        stride_xy=stride_xy,
        radius_fraction=roi_radius_fraction,
    )
    if not xy_positions:
        raise RuntimeError(f"[{state_name}] No valid {patch_size}^2 XY positions inside ROI")

    print(f"[{state_name}] Pass 1/2: scanning {len(files)} slices (load={load_mv:.4f} mV/V)...", flush=True)
    solid_fractions: list[float] = []
    pore_modes: list[float] = []
    ring_scores: list[float] = []
    slice_jumps: list[float] = []
    prev_sub: np.ndarray | None = None

    for idx, path in enumerate(files, start=1):
        img = tifffile.imread(path)
        _, stats = segment_slice_calibrated(
            img, delta_threshold=delta_threshold, roi_mask=roi_mask, sigma=sigma
        )
        solid_fractions.append(stats["solid_fraction"])
        pore_modes.append(stats["pore_mode"])
        ring_scores.append(compute_radial_ring_score(img))

        curr_sub = img[::4, ::4].astype(np.float32)
        slice_jumps.append(0.0 if prev_sub is None else float(np.abs(curr_sub - prev_sub).mean()))
        prev_sub = curr_sub

        if idx % 250 == 0 or idx == len(files):
            print(f"  [{state_name}] Scanned {idx}/{len(files)} slices", flush=True)

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
    split_intervals = split_z_intervals_with_buffer(
        intervals=valid_intervals,
        patch_size=patch_size,
        val_fraction=val_fraction,
        buffer_slices=patch_size,
    )

    patch_dir = output_dir / f"{state_name}_binary_patches{patch_size}_{version}"
    qa_dir = patch_dir / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)

    _save_z_plot(
        out_path=qa_dir / "z_quality_diagnostic.png",
        state_name=state_name,
        solid_fractions=solid_fractions,
        ring_scores=ring_scores,
        slice_jumps=slice_jumps,
        valid_mask=valid_mask,
        intervals=valid_intervals,
    )

    rng = np.random.default_rng(seed)
    val_quota = max(1, int(round(max_patches * val_fraction)))
    quotas = {"train": max_patches - val_quota, "val": val_quota}
    split_counts: dict[str, int] = {}
    split_sf_means: dict[str, float] = {}

    print(f"[{state_name}] Pass 2/2: streaming clean 3D patches -> {patch_dir.name}", flush=True)
    for split_name in ("train", "val"):
        plan = build_stratified_patch_plan(
            z_intervals=split_intervals[split_name],
            xy_positions=xy_positions,
            patch_size=patch_size,
            stride_z=stride_z,
            max_patches=quotas[split_name],
            rng=rng,
        )

        # Sliding ring-buffer cache of segmented slices so overlapping Z windows never re-segment a slice twice
        slice_cache: dict[int, np.ndarray] = {}
        accepted: list[np.ndarray] = []

        for z_start, xy_list in plan:
            if len(accepted) >= quotas[split_name]:
                break
            # Evict slices older than z_start
            for k in [kz for kz in slice_cache if kz < z_start]:
                del slice_cache[k]
            for z_idx in range(z_start, z_start + patch_size):
                if z_idx not in slice_cache:
                    raw_slc = tifffile.imread(files[z_idx])
                    bin_slc, _ = segment_slice_calibrated(
                        raw_slc,
                        delta_threshold=delta_threshold,
                        roi_mask=roi_mask,
                        sigma=sigma,
                    )
                    slice_cache[z_idx] = bin_slc

            slab = np.stack([slice_cache[z_idx] for z_idx in range(z_start, z_start + patch_size)], axis=0)
            for y, x in xy_list:
                if len(accepted) >= quotas[split_name]:
                    break
                patch = np.ascontiguousarray(slab[:, y : y + patch_size, x : x + patch_size], dtype=np.uint8)
                wf = float(patch.mean())
                if min_white <= wf <= max_white:
                    accepted.append(patch)

        if not accepted:
            raise RuntimeError(f"[{state_name}] 0 {split_name} patches passed filter [{min_white}, {max_white}]")

        stacked = np.stack(accepted, axis=0)
        split_out = patch_dir / split_name
        split_out.mkdir(parents=True, exist_ok=True)
        np.save(split_out / "patches.npy", stacked)
        split_counts[split_name] = int(stacked.shape[0])
        split_sf_means[split_name] = float(stacked.mean())
        print(
            f"  [{state_name}] {split_name}: {stacked.shape[0]} patches "
            f"(mean solid={split_sf_means[split_name]:.4f})",
            flush=True,
        )

    sf_arr = np.asarray(solid_fractions, dtype=np.float64)
    clean_sf = sf_arr[valid_mask] if valid_mask.any() else sf_arr

    summary: dict[str, object] = {
        "state": state_name,
        "source_folder": str(scan_dir),
        "load_mv_per_v": load_mv,
        "is_repeat_or_secondary": _is_repeat_or_secondary(state_name),
        "version": version,
        "patch_size": patch_size,
        "input_dtype": str(dtype),
        "slice_shape_hw": list(shape_hw),
        "slices_total": len(files),
        "slices_valid_clean": int(valid_mask.sum()),
        "valid_z_intervals": [[int(s), int(e)] for s, e in valid_intervals],
        "split_z_intervals": {
            k: [[int(s), int(e)] for s, e in v] for k, v in split_intervals.items()
        },
        "clean_solid_fraction_mean": float(clean_sf.mean()),
        "train_patches": split_counts["train"],
        "val_patches": split_counts["val"],
        "train_solid_fraction_mean": split_sf_means["train"],
        "val_solid_fraction_mean": split_sf_means["val"],
    }
    with (qa_dir / "segmentation_report.json").open("w") as f:
        json.dump(summary, f, indent=2)
    with (qa_dir / "valid_slices.json").open("w") as f:
        json.dump(
            {
                "dataset": state_name,
                "load_mv_per_v": load_mv,
                "total_slices": len(files),
                "valid_slice_count": int(valid_mask.sum()),
                "valid_z_intervals": [[int(s), int(e)] for s, e in valid_intervals],
            },
            f,
            indent=2,
        )
    return summary


def build_bundle(
    drive_root: Path,
    output_dir: Path,
    datasets: list[str] | None = None,
    version: str = "carbon_calibrated_v1",
    delta_threshold: float = 5500.0,
    sigma: float = 32.0,
    roi_radius_fraction: float = 0.70,
    patch_size: int = 96,
    stride_xy: int = 96,
    stride_z: int = 32,
    max_patches: int = 1200,
    min_interval_length: int = 96,
    seam_margin: int = 12,
    make_zip: bool = True,
) -> Path:
    if datasets:
        scan_dirs = [drive_root / d for d in datasets]
    else:
        scan_dirs = discover_scan_folders(drive_root)
    if not scan_dirs:
        raise FileNotFoundError(f"No XCT scan folders discovered under {drive_root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    primary_states: dict[str, float] = {}
    repeat_states: dict[str, float] = {}
    scan_summaries: list[dict[str, object]] = []

    for scan_dir in scan_dirs:
        summary = stream_bundle_single_scan(
            scan_dir=scan_dir,
            output_dir=output_dir,
            version=version,
            delta_threshold=delta_threshold,
            sigma=sigma,
            roi_radius_fraction=roi_radius_fraction,
            patch_size=patch_size,
            stride_xy=stride_xy,
            stride_z=stride_z,
            max_patches=max_patches,
            min_interval_length=min_interval_length,
            seam_margin=seam_margin,
        )
        scan_summaries.append(summary)
        st_name = str(summary["state"])
        mv = float(summary["load_mv_per_v"])  # type: ignore[arg-type]
        if summary["is_repeat_or_secondary"]:
            repeat_states[st_name] = mv
        else:
            primary_states[st_name] = mv

    # Write direct `--states-json` file for primary series + full manifest
    with (output_dir / "primary_states.json").open("w") as f:
        json.dump(dict(sorted(primary_states.items(), key=lambda kv: kv[1])), f, indent=2)

    manifest = {
        "version": version,
        "patch_size": patch_size,
        "delta_threshold": delta_threshold,
        "primary_states": dict(sorted(primary_states.items(), key=lambda kv: kv[1])),
        "repeat_and_secondary_states": dict(sorted(repeat_states.items(), key=lambda kv: kv[1])),
        "scans": scan_summaries,
    }
    manifest_path = output_dir / "bundle_manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    if make_zip:
        zip_path = output_dir.parent / f"{output_dir.name}.zip"
        print(f"Compressing bundle -> {zip_path} ...", flush=True)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for file_path in sorted(output_dir.rglob("*")):
                if file_path.is_file():
                    zf.write(file_path, arcname=file_path.relative_to(output_dir))
        size_mb = zip_path.stat().st_size / (1024**2)
        print(f"Bundle ready: {zip_path} ({size_mb:.1f} MiB)", flush=True)
        return zip_path

    return manifest_path


def main() -> None:
    # Support zero-argument F5 execution directly inside Spyder on Windows
    if len(sys.argv) == 1 and SPYDER_DRIVE_ROOT.exists():
        build_bundle(
            drive_root=SPYDER_DRIVE_ROOT,
            output_dir=SPYDER_OUTPUT_DIR,
            make_zip=True,
        )
        return

    parser = argparse.ArgumentParser(
        description="Stream-segment raw XCT scans on a local drive directly into uploadable 3D patch bundles"
    )
    parser.add_argument("--drive-root", type=Path, required=True, help="Root folder (e.g. D:/dry_samples)")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output bundle directory")
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional subset of scan folders")
    parser.add_argument("--version", default="carbon_calibrated_v1")
    parser.add_argument("--delta-threshold", type=float, default=5500.0)
    parser.add_argument("--sigma", type=float, default=32.0)
    parser.add_argument("--roi-radius-fraction", type=float, default=0.70)
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--stride-xy", type=int, default=96)
    parser.add_argument("--stride-z", type=int, default=32)
    parser.add_argument("--max-patches", type=int, default=1200)
    parser.add_argument("--min-interval-length", type=int, default=96)
    parser.add_argument("--seam-margin", type=int, default=12)
    parser.add_argument("--zip", action="store_true", help="Compress output directory into a single .zip archive")
    args = parser.parse_args()

    build_bundle(
        drive_root=args.drive_root,
        output_dir=args.output_dir,
        datasets=args.datasets,
        version=args.version,
        delta_threshold=args.delta_threshold,
        sigma=args.sigma,
        roi_radius_fraction=args.roi_radius_fraction,
        patch_size=args.patch_size,
        stride_xy=args.stride_xy,
        stride_z=args.stride_z,
        max_patches=args.max_patches,
        min_interval_length=args.min_interval_length,
        seam_margin=args.seam_margin,
        make_zip=args.zip,
    )


if __name__ == "__main__":
    main()
