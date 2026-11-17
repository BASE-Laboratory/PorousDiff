"""Stream-segment raw XCT scans (Virgin, TiO2 Anode, CuHCF Cathode, DIAD) into 3D patch bundles.

Supports:
  1. 10-second `--preview-only` mode generating `qa/multiphase_overlay_diagnostic.png` (`pore=black`,
     `carbon=cyan`, `active_coating=gold`) so 3-phase thresholds can be visually verified in Spyder.
  2. Automatic material inference (`virgin_carbon`, `tio2_anode`, `cuhcf_cathode`), 2-phase or
     3-phase pore-referenced segmentation, and bullseye/plunger Z-interval gating.
  3. Topology-preserving `--spatial-bin` (`1x` native or `2x` REV-scale block binning) + physical
     calibration metadata (`voxel_size_um`, `stress_kpa`, `strain_z`, `beamline`).
  4. Single `.zip` archive output (`--zip`) with `primary_states.json` + `bundle_manifest.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from matplotlib.colors import ListedColormap
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
    CalibrationMetadata,
    build_stratified_patch_plan,
    calibrate_delta_threshold,
    calibrate_multiphase_thresholds,
    compute_pore_referenced_delta,
    compute_radial_ring_score,
    detect_valid_z_intervals,
    downsample_volume_rev,
    enumerate_cylinder_xy_positions,
    infer_material_name,
    load_overrides_table,
    make_circular_mask,
    make_collision_safe_state_name,
    parse_load_with_unit,
    resolve_tiff_directory,
    segment_slice_multiphase,
    split_z_intervals_with_buffer,
)

SPYDER_DRIVE_ROOT = Path("D:/dry_samples")
SPYDER_OUTPUT_DIR = Path("D:/porousdiff_upload_bundle")
PHASE_CMAP = ListedColormap(["#0b0e14", "#38bdf8", "#facc15"])


def _is_repeat_or_secondary(folder_or_path: str) -> bool:
    low = folder_or_path.lower()
    return ("rep" in low) or ("virgin_2" in low) or ("sample2" in low) or ("diad" in low and "rep" in low)


def _save_multiphase_preview(
    out_path: Path,
    state_name: str,
    material: str,
    raw_slice: np.ndarray,
    seg_slice: np.ndarray,
    delta_slice: np.ndarray,
    roi_mask: np.ndarray,
    delta_carbon: float,
    delta_coating: float | None,
) -> None:
    h, w = raw_slice.shape
    ch = min(h, max(96, h // 4))
    cw = min(w, max(96, w // 4))
    y0, x0 = (h - ch) // 2, (w - cw) // 2

    raw_crop = raw_slice[y0 : y0 + ch, x0 : x0 + cw].astype(np.float32)
    seg_crop = seg_slice[y0 : y0 + ch, x0 : x0 + cw]
    vmin, vmax = np.percentile(raw_crop, [1.0, 99.5])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    axes[0].imshow(raw_crop, cmap="gray", vmin=vmin, vmax=max(vmax, vmin + 1.0))
    axes[0].set_title(f"{state_name} ({material})\nRaw Center Crop ({ch}x{cw})")
    axes[0].axis("off")

    axes[1].imshow(seg_crop, cmap=PHASE_CMAP, vmin=0, vmax=2, interpolation="nearest")
    c_frac = float((seg_slice[roi_mask] == 1).mean())
    a_frac = float((seg_slice[roi_mask] == 2).mean())
    axes[1].set_title(
        f"Segmented (Black=Pore, Cyan=Carbon {c_frac:.3f}, Gold=Coating {a_frac:.3f})"
    )
    axes[1].axis("off")

    roi_delta = delta_slice[roi_mask]
    lo, hi = np.percentile(roi_delta, [0.5, 99.9])
    clipped = roi_delta[(roi_delta >= lo) & (roi_delta <= hi)]
    axes[2].hist(clipped, bins=120, color="slategray", log=True, alpha=0.85)
    axes[2].axvline(delta_carbon, color="#0284c7", lw=1.8, label=f"delta_carbon={delta_carbon:.0f}")
    if delta_coating is not None:
        axes[2].axvline(
            delta_coating, color="#eab308", lw=1.8, label=f"delta_coating={delta_coating:.0f}"
        )
    axes[2].set_xlabel("Pore-referenced intensity delta")
    axes[2].set_ylabel("Voxel count (log)")
    axes[2].set_title("Phase Threshold Calibration")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_z_plot(
    out_path: Path,
    state_name: str,
    solid_fractions: list[float],
    coating_fractions: list[float],
    ring_scores: list[float],
    slice_jumps: list[float],
    valid_mask: np.ndarray,
    intervals: list[tuple[int, int]],
) -> None:
    z = np.arange(len(solid_fractions))
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(z, solid_fractions, color="tab:blue", lw=1.3, label="Total solid fraction")
    if any(c > 0.0 for c in coating_fractions):
        axes[0].plot(z, coating_fractions, color="goldenrod", lw=1.2, label="Active coating fraction")
    for idx, (s, e) in enumerate(intervals):
        axes[0].axvspan(
            s, e - 1, color="tab:green", alpha=0.15, label="Accepted clean Z" if idx == 0 else None
        )
    axes[0].set_ylabel("Volume fraction")
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


def _clean_ref_indices(num_files: int) -> list[int]:
    """Select clean upper-middle Z slice indices (`20%`, `32%`, `45%`) avoiding top edge & bottom plunger."""
    if num_files <= 3:
        return list(range(num_files))
    raw = [int(0.20 * num_files), int(0.32 * num_files), int(0.45 * num_files)]
    return sorted({min(max(0, i), num_files - 1) for i in raw})


def discover_scan_folders(
    drive_root: Path,
    max_depth: int = 3,
    overrides: dict[str, tuple[float, str]] | None = None,
) -> list[Path]:
    """Recursively discover XCT scan folders up to `max_depth` levels under `drive_root`."""
    candidates: list[Path] = []

    def _walk(curr: Path, depth: int) -> None:
        if depth > max_depth or not curr.is_dir():
            return
        for entry in sorted(curr.iterdir()):
            if not entry.is_dir() or entry.name.startswith(".") or entry.name in ("qa", "rec_16bit_Paganin"):
                continue
            rel_key = str(entry.relative_to(drive_root)).replace("\\", "/")
            has_tiffs = False
            try:
                resolve_tiff_directory(entry)
                has_tiffs = True
            except FileNotFoundError:
                has_tiffs = False

            if has_tiffs:
                try:
                    parse_load_with_unit(rel_key, overrides=overrides)
                    candidates.append(entry)
                    continue
                except ValueError:
                    pass
            _walk(entry, depth + 1)

    _walk(drive_root, 1)
    return candidates


def stream_bundle_single_scan(
    scan_dir: Path,
    output_dir: Path,
    state_name: str | None = None,
    version: str = "carbon_calibrated_v1",
    material: str = "auto",
    num_phases: int = 2,
    auto_multiphase: bool = False,
    auto_calibrate_carbon: bool = False,
    delta_threshold: float = 5500.0,
    delta_coating_threshold: float = 0.0,
    precalibrated_thresholds: tuple[float, float | None] | None = None,
    load_overrides: dict[str, tuple[float, str]] | None = None,
    sigma: float = 32.0,
    roi_radius_fraction: float = 0.70,
    patch_size: int = 96,
    spatial_bin: int = 1,
    stride_xy: int = 96,
    stride_z: int = 32,
    max_patches: int = 1200,
    min_white: float = 0.005,
    max_white: float = 0.90,
    val_fraction: float = 0.2,
    min_interval_length: int = 96,
    max_ring_score: float = 0.15,
    seam_margin: int = 12,
    calibration: CalibrationMetadata | None = None,
    preview_only: bool = False,
    resume: bool = False,
    seed: int = 42,
) -> dict[str, object]:
    calib = calibration or CalibrationMetadata(spatial_bin=spatial_bin)
    resolved_material = infer_material_name(scan_dir) if material == "auto" else material
    resolved_state = state_name or make_collision_safe_state_name(
        scan_dir, material=resolved_material, beamline=calib.beamline
    )
    load_val, load_unit = parse_load_with_unit(str(scan_dir), overrides=load_overrides)
    stress_kpa = calib.resolve_stress_kpa(load_val, unit=load_unit)
    load_mv = load_val if load_unit == "mv_per_v" else (
        (stress_kpa / calib.kpa_per_mvv)
        if (stress_kpa is not None and calib.kpa_per_mvv)
        else load_val
    )

    use_three_phase = (num_phases == 3) or (auto_multiphase and resolved_material != "virgin_carbon")
    eff_num_phases = 3 if (use_three_phase and resolved_material != "virgin_carbon") else 2

    patch_dir = output_dir / f"{resolved_state}_binary_patches{patch_size}_{version}"
    qa_dir = patch_dir / "qa"
    report_path = qa_dir / "segmentation_report.json"

    if (
        resume
        and not preview_only
        and (patch_dir / "train" / "patches.npy").is_file()
        and (patch_dir / "val" / "patches.npy").is_file()
        and report_path.is_file()
    ):
        cached = json.loads(report_path.read_text())
        print(f"[{resolved_state}] Resuming: reusing existing patches in {patch_dir.name}", flush=True)
        return cached

    tiff_dir = resolve_tiff_directory(scan_dir)
    files = sorted(list(tiff_dir.glob("*.tif")) + list(tiff_dir.glob("*.tiff")))
    raw_window = patch_size * max(1, spatial_bin)
    if not preview_only and len(files) < 2 * raw_window:
        raise RuntimeError(
            f"[{resolved_state}] Need >= {2 * raw_window} slices in {tiff_dir}, found {len(files)}"
        )

    with tifffile.TiffFile(files[0]) as first_tif:
        shape_hw = tuple(first_tif.pages[0].shape)
        dtype = first_tif.pages[0].dtype

    roi_mask = make_circular_mask(shape_hw, radius_fraction=roi_radius_fraction)

    if precalibrated_thresholds is not None:
        eff_delta_carbon, delta_coating = precalibrated_thresholds
    else:
        ref_slices = [tifffile.imread(files[i]) for i in _clean_ref_indices(len(files))]
        if eff_num_phases == 3:
            eff_delta_carbon, auto_coat = calibrate_multiphase_thresholds(
                ref_slices,
                roi_mask=roi_mask,
                sigma=sigma,
                fixed_delta_carbon=None if auto_calibrate_carbon else delta_threshold,
            )
            delta_coating = (
                float(delta_coating_threshold)
                if delta_coating_threshold > eff_delta_carbon
                else auto_coat
            )
        else:
            eff_delta_carbon = (
                calibrate_delta_threshold(ref_slices, roi_mask=roi_mask, sigma=sigma)
                if auto_calibrate_carbon
                else float(delta_threshold)
            )
            delta_coating = None

    qa_dir.mkdir(parents=True, exist_ok=True)

    mid_idx = int(0.35 * len(files)) if len(files) >= 16 else (len(files) // 2)
    mid_raw = tifffile.imread(files[mid_idx])
    mid_delta, _, _ = compute_pore_referenced_delta(mid_raw, roi_mask=roi_mask, sigma=sigma)
    mid_seg, mid_stats = segment_slice_multiphase(
        mid_raw,
        delta_carbon=eff_delta_carbon,
        delta_coating=delta_coating,
        roi_mask=roi_mask,
        sigma=sigma,
    )
    _save_multiphase_preview(
        out_path=qa_dir / "multiphase_overlay_diagnostic.png",
        state_name=resolved_state,
        material=resolved_material,
        raw_slice=mid_raw,
        seg_slice=mid_seg,
        delta_slice=mid_delta,
        roi_mask=roi_mask,
        delta_carbon=eff_delta_carbon,
        delta_coating=delta_coating,
    )

    if preview_only:
        preview_info: dict[str, object] = {
            "state": resolved_state,
            "material": resolved_material,
            "beamline": calib.beamline,
            "load_mv_per_v": load_mv,
            "load_raw_value": load_val,
            "load_unit": load_unit,
            "stress_kpa": stress_kpa,
            "effective_voxel_size_um": calib.effective_voxel_size_um,
            "num_phases": eff_num_phases,
            "delta_carbon": eff_delta_carbon,
            "delta_coating": delta_coating,
            "mid_slice_stats": mid_stats,
            "preview_figure": str(qa_dir / "multiphase_overlay_diagnostic.png"),
        }
        with (qa_dir / "preview_summary.json").open("w") as f:
            json.dump(preview_info, f, indent=2)
        print(f"[{resolved_state}] Saved preview -> {qa_dir / 'multiphase_overlay_diagnostic.png'}", flush=True)
        return preview_info

    raw_stride_xy = stride_xy * max(1, spatial_bin)
    raw_stride_z = stride_z * max(1, spatial_bin)
    xy_positions = enumerate_cylinder_xy_positions(
        shape_hw=shape_hw,
        patch_size=raw_window,
        stride_xy=raw_stride_xy,
        radius_fraction=roi_radius_fraction,
    )
    if not xy_positions:
        raise RuntimeError(f"[{resolved_state}] No valid {raw_window}^2 XY positions inside ROI")

    print(
        f"[{resolved_state}] Pass 1/2: scanning {len(files)} slices "
        f"(material={resolved_material}, phases={eff_num_phases}, load={load_val:.4f} {load_unit})...",
        flush=True,
    )
    solid_fractions: list[float] = []
    coating_fractions: list[float] = []
    pore_modes: list[float] = []
    ring_scores: list[float] = []
    slice_jumps: list[float] = []
    prev_sub: np.ndarray | None = None

    for idx, path in enumerate(files, start=1):
        img = tifffile.imread(path)
        _, stats = segment_slice_multiphase(
            img,
            delta_carbon=eff_delta_carbon,
            delta_coating=delta_coating,
            roi_mask=roi_mask,
            sigma=sigma,
        )
        solid_fractions.append(stats["solid_fraction"])
        coating_fractions.append(stats["coating_fraction"])
        pore_modes.append(stats["pore_mode"])
        ring_scores.append(compute_radial_ring_score(img))

        curr_sub = img[::4, ::4].astype(np.float32)
        slice_jumps.append(0.0 if prev_sub is None else float(np.abs(curr_sub - prev_sub).mean()))
        prev_sub = curr_sub

        if idx % 250 == 0 or idx == len(files):
            print(f"  [{resolved_state}] Scanned {idx}/{len(files)} slices", flush=True)

    if len(slice_jumps) > 1:
        slice_jumps[0] = slice_jumps[1]

    eff_min_interval = max(min_interval_length, raw_window)
    valid_mask, valid_intervals = detect_valid_z_intervals(
        ring_scores=ring_scores,
        pore_modes=pore_modes,
        slice_jumps=slice_jumps,
        min_interval_length=eff_min_interval,
        max_ring_score=max_ring_score,
        seam_margin=seam_margin,
    )
    split_intervals = split_z_intervals_with_buffer(
        intervals=valid_intervals,
        patch_size=raw_window,
        val_fraction=val_fraction,
        buffer_slices=raw_window,
    )

    _save_z_plot(
        out_path=qa_dir / "z_quality_diagnostic.png",
        state_name=resolved_state,
        solid_fractions=solid_fractions,
        coating_fractions=coating_fractions,
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
    split_coat_means: dict[str, float] = {}

    print(f"[{resolved_state}] Pass 2/2: streaming clean 3D patches -> {patch_dir.name}", flush=True)
    for split_name in ("train", "val"):
        plan = build_stratified_patch_plan(
            z_intervals=split_intervals[split_name],
            xy_positions=xy_positions,
            patch_size=raw_window,
            stride_z=raw_stride_z,
            max_patches=quotas[split_name],
            rng=rng,
        )

        slice_cache: dict[int, np.ndarray] = {}
        accepted: list[np.ndarray] = []

        for z_start, xy_list in plan:
            if len(accepted) >= quotas[split_name]:
                break
            for k in [kz for kz in slice_cache if kz < z_start]:
                del slice_cache[k]
            for z_idx in range(z_start, z_start + raw_window):
                if z_idx not in slice_cache:
                    raw_slc = tifffile.imread(files[z_idx])
                    seg_slc, _ = segment_slice_multiphase(
                        raw_slc,
                        delta_carbon=eff_delta_carbon,
                        delta_coating=delta_coating,
                        roi_mask=roi_mask,
                        sigma=sigma,
                    )
                    slice_cache[z_idx] = seg_slc

            slab = np.stack([slice_cache[z_idx] for z_idx in range(z_start, z_start + raw_window)], axis=0)
            for y, x in xy_list:
                if len(accepted) >= quotas[split_name]:
                    break
                raw_patch = np.ascontiguousarray(
                    slab[:, y : y + raw_window, x : x + raw_window], dtype=np.uint8
                )
                patch = downsample_volume_rev(raw_patch, factor=spatial_bin)
                wf = float((patch >= 1).mean())
                if min_white <= wf <= max_white:
                    accepted.append(patch)

        if not accepted:
            raise RuntimeError(f"[{resolved_state}] 0 {split_name} patches passed filter [{min_white}, {max_white}]")

        stacked = np.stack(accepted, axis=0)
        split_out = patch_dir / split_name
        split_out.mkdir(parents=True, exist_ok=True)
        np.save(split_out / "patches.npy", stacked)
        split_counts[split_name] = int(stacked.shape[0])
        split_sf_means[split_name] = float((stacked >= 1).mean())
        split_coat_means[split_name] = float((stacked == 2).mean())
        print(
            f"  [{resolved_state}] {split_name}: {stacked.shape[0]} patches "
            f"(solid={split_sf_means[split_name]:.4f}, coating={split_coat_means[split_name]:.4f})",
            flush=True,
        )

    sf_arr = np.asarray(solid_fractions, dtype=np.float64)
    cf_arr = np.asarray(coating_fractions, dtype=np.float64)
    clean_sf = sf_arr[valid_mask] if valid_mask.any() else sf_arr
    clean_cf = cf_arr[valid_mask] if valid_mask.any() else cf_arr
    clean_slices = int(valid_mask.sum())

    summary: dict[str, object] = {
        "state": resolved_state,
        "material": resolved_material,
        "beamline": calib.beamline,
        "source_folder": str(scan_dir),
        "patch_dir_name": patch_dir.name,
        "load_mv_per_v": load_mv,
        "load_raw_value": load_val,
        "load_unit": load_unit,
        "stress_kpa": stress_kpa,
        "strain_z": calib.estimate_strain_z(clean_slices),
        "voxel_size_um": calib.voxel_size_um,
        "spatial_bin": spatial_bin,
        "effective_voxel_size_um": calib.effective_voxel_size_um,
        "num_phases": eff_num_phases,
        "delta_carbon": eff_delta_carbon,
        "delta_coating": delta_coating,
        "is_repeat_or_secondary": _is_repeat_or_secondary(resolved_state),
        "version": version,
        "patch_size": patch_size,
        "input_dtype": str(dtype),
        "slice_shape_hw": list(shape_hw),
        "slices_total": len(files),
        "slices_valid_clean": clean_slices,
        "valid_z_intervals": [[int(s), int(e)] for s, e in valid_intervals],
        "split_z_intervals": {
            k: [[int(s), int(e)] for s, e in v] for k, v in split_intervals.items()
        },
        "clean_solid_fraction_mean": float(clean_sf.mean()),
        "clean_coating_fraction_mean": float(clean_cf.mean()),
        "train_patches": split_counts["train"],
        "val_patches": split_counts["val"],
        "train_solid_fraction_mean": split_sf_means["train"],
        "val_solid_fraction_mean": split_sf_means["val"],
        "train_coating_fraction_mean": split_coat_means["train"],
        "val_coating_fraction_mean": split_coat_means["val"],
    }
    with (qa_dir / "segmentation_report.json").open("w") as f:
        json.dump(summary, f, indent=2)
    with (qa_dir / "valid_slices.json").open("w") as f:
        json.dump(
            {
                "dataset": resolved_state,
                "material": resolved_material,
                "load_mv_per_v": load_mv,
                "stress_kpa": stress_kpa,
                "total_slices": len(files),
                "valid_slice_count": clean_slices,
                "valid_z_intervals": [[int(s), int(e)] for s, e in valid_intervals],
            },
            f,
            indent=2,
        )
    return summary


def _calibrate_group_thresholds(
    ref_scan_dir: Path,
    eff_num_phases: int,
    auto_calibrate_carbon: bool,
    delta_threshold: float,
    delta_coating_threshold: float,
    roi_radius_fraction: float,
    sigma: float,
) -> tuple[float, float | None]:
    """Calibrate one shared `(delta_carbon, delta_coating)` pair on a group's lowest-load scan."""
    tiff_dir = resolve_tiff_directory(ref_scan_dir)
    files = sorted(list(tiff_dir.glob("*.tif")) + list(tiff_dir.glob("*.tiff")))
    with tifffile.TiffFile(files[0]) as first_tif:
        shape_hw = tuple(first_tif.pages[0].shape)
    roi_mask = make_circular_mask(shape_hw, radius_fraction=roi_radius_fraction)
    ref_slices = [tifffile.imread(files[i]) for i in _clean_ref_indices(len(files))]

    if eff_num_phases == 3:
        d_carb, auto_coat = calibrate_multiphase_thresholds(
            ref_slices,
            roi_mask=roi_mask,
            sigma=sigma,
            fixed_delta_carbon=None if auto_calibrate_carbon else delta_threshold,
        )
        d_coat = (
            float(delta_coating_threshold)
            if delta_coating_threshold > d_carb
            else auto_coat
        )
        return float(d_carb), float(d_coat)

    d_carb = (
        calibrate_delta_threshold(ref_slices, roi_mask=roi_mask, sigma=sigma)
        if auto_calibrate_carbon
        else float(delta_threshold)
    )
    return float(d_carb), None


def build_bundle(
    drive_root: Path,
    output_dir: Path,
    datasets: list[str] | None = None,
    version: str = "carbon_calibrated_v1",
    material: str = "auto",
    num_phases: int = 2,
    auto_multiphase: bool = False,
    auto_calibrate_carbon: bool = False,
    delta_threshold: float = 5500.0,
    delta_coating_threshold: float = 0.0,
    loads_csv: Path | None = None,
    sigma: float = 32.0,
    roi_radius_fraction: float = 0.70,
    patch_size: int = 96,
    spatial_bin: int = 1,
    stride_xy: int = 96,
    stride_z: int = 32,
    max_patches: int = 1200,
    min_interval_length: int = 96,
    seam_margin: int = 12,
    voxel_size_um: float = 1.0,
    kpa_per_mvv: float | None = None,
    punch_diameter_mm: float | None = None,
    beamline: str = "primary",
    preview_only: bool = False,
    append: bool = False,
    resume: bool = False,
    make_zip: bool = True,
) -> Path:
    overrides = load_overrides_table(loads_csv) if loads_csv is not None else None
    scan_dirs = (
        [drive_root / d for d in datasets]
        if datasets
        else discover_scan_folders(drive_root, overrides=overrides)
    )
    if not scan_dirs:
        raise FileNotFoundError(f"No XCT scan folders discovered under {drive_root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    existing_scans_by_state: dict[str, dict[str, object]] = {}
    manifest_path = output_dir / "bundle_manifest.json"
    if append and not preview_only and manifest_path.is_file():
        prev_manifest = json.loads(manifest_path.read_text())
        for prev_sc in prev_manifest.get("scans", []):
            existing_scans_by_state[str(prev_sc["state"])] = dict(prev_sc)

    # Plan canonical state names and group by (material, beamline) sorted by ascending load
    seen_names: set[str] = set()
    planned: list[dict[str, object]] = []
    for scan_dir in scan_dirs:
        try:
            rel_key = str(scan_dir.relative_to(drive_root)).replace("\\", "/")
        except ValueError:
            rel_key = str(scan_dir)
        res_mat = infer_material_name(rel_key) if material == "auto" else material
        st_name = make_collision_safe_state_name(
            scan_dir, material=res_mat, beamline=beamline, existing_names=seen_names
        )
        seen_names.add(st_name)
        lv, lu = parse_load_with_unit(rel_key, overrides=overrides)
        planned.append(
            {
                "scan_dir": scan_dir,
                "state_name": st_name,
                "material": res_mat,
                "load_val": lv,
                "load_unit": lu,
            }
        )

    planned.sort(key=lambda item: (str(item["material"]), float(item["load_val"])))  # type: ignore[arg-type]

    # Calibrate ONE shared (delta_carbon, delta_coating) pair per (material, beamline) from lowest-load scan
    group_thresholds: dict[tuple[str, str], tuple[float, float | None]] = {}
    for item in planned:
        res_mat = str(item["material"])
        grp_key = (res_mat, beamline)
        if grp_key in group_thresholds:
            continue
        use_3p = (num_phases == 3) or (auto_multiphase and res_mat != "virgin_carbon")
        eff_nph = 3 if (use_3p and res_mat != "virgin_carbon") else 2
        group_thresholds[grp_key] = _calibrate_group_thresholds(
            ref_scan_dir=item["scan_dir"],  # type: ignore[arg-type]
            eff_num_phases=eff_nph,
            auto_calibrate_carbon=auto_calibrate_carbon,
            delta_threshold=delta_threshold,
            delta_coating_threshold=delta_coating_threshold,
            roi_radius_fraction=roi_radius_fraction,
            sigma=sigma,
        )

    ref_thickness_by_group: dict[tuple[str, str], int] = {}
    new_summaries: list[dict[str, object]] = []

    for item in planned:
        res_mat = str(item["material"])
        grp_key = (res_mat, beamline)
        calib = CalibrationMetadata(
            voxel_size_um=voxel_size_um,
            spatial_bin=spatial_bin,
            kpa_per_mvv=kpa_per_mvv,
            punch_diameter_mm=punch_diameter_mm,
            reference_thickness_slices=ref_thickness_by_group.get(grp_key),
            beamline=beamline,
        )
        summary = stream_bundle_single_scan(
            scan_dir=item["scan_dir"],  # type: ignore[arg-type]
            output_dir=output_dir,
            state_name=str(item["state_name"]),
            version=version,
            material=res_mat,
            num_phases=num_phases,
            auto_multiphase=auto_multiphase,
            auto_calibrate_carbon=auto_calibrate_carbon,
            delta_threshold=delta_threshold,
            delta_coating_threshold=delta_coating_threshold,
            precalibrated_thresholds=group_thresholds[grp_key],
            load_overrides=overrides,
            sigma=sigma,
            roi_radius_fraction=roi_radius_fraction,
            patch_size=patch_size,
            spatial_bin=spatial_bin,
            stride_xy=stride_xy,
            stride_z=stride_z,
            max_patches=max_patches,
            min_interval_length=min_interval_length,
            seam_margin=seam_margin,
            calibration=calib,
            preview_only=preview_only,
            resume=resume,
        )
        if not preview_only and grp_key not in ref_thickness_by_group:
            ref_slices = int(summary["slices_valid_clean"])  # type: ignore[arg-type]
            ref_thickness_by_group[grp_key] = ref_slices
            summary["strain_z"] = 0.0
            qa_rep = output_dir / str(summary["patch_dir_name"]) / "qa" / "segmentation_report.json"
            if qa_rep.is_file():
                with qa_rep.open("w") as f:
                    json.dump(summary, f, indent=2)

        new_summaries.append(summary)

    if preview_only:
        preview_manifest = output_dir / "preview_manifest.json"
        with preview_manifest.open("w") as f:
            json.dump({"previews": new_summaries}, f, indent=2)
        return preview_manifest

    for summary in new_summaries:
        existing_scans_by_state[str(summary["state"])] = summary
    scan_summaries = list(existing_scans_by_state.values())

    primary_states: dict[str, float] = {}
    repeat_states: dict[str, float] = {}
    by_material: dict[str, dict[str, float]] = {}
    for summary in scan_summaries:
        st_name = str(summary["state"])
        mat_name = str(summary["material"])
        mv = float(summary["load_mv_per_v"])  # type: ignore[arg-type]
        if summary["is_repeat_or_secondary"]:
            repeat_states[st_name] = mv
        else:
            primary_states[st_name] = mv
            by_material.setdefault(mat_name, {})[st_name] = mv

    with (output_dir / "primary_states.json").open("w") as f:
        json.dump(dict(sorted(primary_states.items(), key=lambda kv: kv[1])), f, indent=2)

    manifest = {
        "version": version,
        "patch_size": patch_size,
        "spatial_bin": spatial_bin,
        "voxel_size_um": voxel_size_um,
        "effective_voxel_size_um": float(voxel_size_um * max(1, spatial_bin)),
        "kpa_per_mvv": kpa_per_mvv,
        "punch_diameter_mm": punch_diameter_mm,
        "beamline": beamline,
        "delta_threshold": delta_threshold,
        "primary_states": dict(sorted(primary_states.items(), key=lambda kv: kv[1])),
        "states_by_material": {
            m: dict(sorted(d.items(), key=lambda kv: kv[1])) for m, d in sorted(by_material.items())
        },
        "repeat_and_secondary_states": dict(sorted(repeat_states.items(), key=lambda kv: kv[1])),
        "scans": scan_summaries,
    }
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
    if len(sys.argv) == 1 and SPYDER_DRIVE_ROOT.exists():
        build_bundle(
            drive_root=SPYDER_DRIVE_ROOT,
            output_dir=SPYDER_OUTPUT_DIR,
            auto_multiphase=True,
            spatial_bin=2,
            resume=True,
            make_zip=True,
        )
        return

    parser = argparse.ArgumentParser(
        description="Stream-segment raw multi-electrode XCT scans directly into uploadable 3D patch bundles"
    )
    parser.add_argument("--drive-root", type=Path, required=True, help="Root folder (e.g. D:/dry_samples)")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output bundle directory")
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional subset of scan folders")
    parser.add_argument("--version", default="carbon_calibrated_v1")
    parser.add_argument(
        "--material",
        default="auto",
        choices=["auto", "virgin_carbon", "tio2_anode", "cuhcf_cathode"],
    )
    parser.add_argument("--num-phases", type=int, default=2, choices=[2, 3])
    parser.add_argument(
        "--auto-multiphase",
        action="store_true",
        help="Automatically segment TiO2 and CuHCF scans into 3 phases {pore, carbon, coating} and Virgin into 2 phases",
    )
    parser.add_argument(
        "--auto-calibrate-carbon",
        action="store_true",
        help="Auto-calibrate delta_carbon per (material, beamline) from lowest-load scan (recommended for K11 DIAD)",
    )
    parser.add_argument("--delta-threshold", type=float, default=5500.0)
    parser.add_argument(
        "--delta-coating-threshold",
        type=float,
        default=0.0,
        help="Explicit coating delta threshold (> delta-threshold); 0 auto-calibrates via 2-stage solid Otsu",
    )
    parser.add_argument(
        "--loads-csv",
        type=Path,
        default=None,
        help="Optional CSV/JSON lookup table mapping folder names to load values/units",
    )
    parser.add_argument("--sigma", type=float, default=32.0)
    parser.add_argument("--roi-radius-fraction", type=float, default=0.70)
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--spatial-bin", type=int, default=1, help="Spatial block-binning factor (e.g. 2 for REV scale)")
    parser.add_argument("--stride-xy", type=int, default=96)
    parser.add_argument("--stride-z", type=int, default=32)
    parser.add_argument("--max-patches", type=int, default=1200)
    parser.add_argument("--min-interval-length", type=int, default=96)
    parser.add_argument("--seam-margin", type=int, default=12)
    parser.add_argument("--voxel-size-um", type=float, default=1.0)
    parser.add_argument("--kpa-per-mvv", type=float, default=None)
    parser.add_argument("--punch-diameter-mm", type=float, default=None)
    parser.add_argument("--beamline", default="primary")
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Fast 10s check: save multiphase_overlay_diagnostic.png without extracting 3D patches",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Merge newly bundled scans into an existing bundle_manifest.json (for multi-drive / K11 DIAD workflows)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip scans whose train/val patches.npy and segmentation_report.json already exist",
    )
    parser.add_argument("--zip", action="store_true", help="Compress output directory into a single .zip archive")
    args = parser.parse_args()

    build_bundle(
        drive_root=args.drive_root,
        output_dir=args.output_dir,
        datasets=args.datasets,
        version=args.version,
        material=args.material,
        num_phases=args.num_phases,
        auto_multiphase=args.auto_multiphase,
        auto_calibrate_carbon=args.auto_calibrate_carbon,
        delta_threshold=args.delta_threshold,
        delta_coating_threshold=args.delta_coating_threshold,
        loads_csv=args.loads_csv,
        sigma=args.sigma,
        roi_radius_fraction=args.roi_radius_fraction,
        patch_size=args.patch_size,
        spatial_bin=args.spatial_bin,
        stride_xy=args.stride_xy,
        stride_z=args.stride_z,
        max_patches=args.max_patches,
        min_interval_length=args.min_interval_length,
        seam_margin=args.seam_margin,
        voxel_size_um=args.voxel_size_um,
        kpa_per_mvv=args.kpa_per_mvv,
        punch_diameter_mm=args.punch_diameter_mm,
        beamline=args.beamline,
        preview_only=args.preview_only,
        append=args.append,
        resume=args.resume,
        make_zip=args.zip,
    )


if __name__ == "__main__":
    main()
