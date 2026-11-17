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
    calibrate_multiphase_thresholds,
    compute_pore_referenced_delta,
    compute_radial_ring_score,
    detect_valid_z_intervals,
    downsample_volume_rev,
    enumerate_cylinder_xy_positions,
    infer_material_name,
    make_circular_mask,
    parse_load_value,
    resolve_tiff_directory,
    segment_slice_multiphase,
    split_z_intervals_with_buffer,
)

SPYDER_DRIVE_ROOT = Path("D:/dry_samples")
SPYDER_OUTPUT_DIR = Path("D:/porousdiff_upload_bundle")
PHASE_CMAP = ListedColormap(["#0b0e14", "#38bdf8", "#facc15"])


def _clean_state_name(folder_name: str) -> str:
    return folder_name.strip().rstrip("_")


def _is_repeat_or_secondary(folder_name: str) -> bool:
    low = folder_name.lower()
    return ("rep" in low) or ("virgin_2" in low) or ("sample2" in low)


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
    material: str = "auto",
    num_phases: int = 2,
    auto_multiphase: bool = False,
    delta_threshold: float = 5500.0,
    delta_coating_threshold: float = 0.0,
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
    seed: int = 42,
) -> dict[str, object]:
    calib = calibration or CalibrationMetadata(spatial_bin=spatial_bin)
    state_name = _clean_state_name(scan_dir.name)
    resolved_material = infer_material_name(state_name) if material == "auto" else material
    load_mv = parse_load_value(state_name)
    stress_kpa = calib.load_to_stress_kpa(load_mv)

    use_three_phase = (num_phases == 3) or (auto_multiphase and resolved_material != "virgin_carbon")
    eff_num_phases = 3 if (use_three_phase and resolved_material != "virgin_carbon") else 2

    tiff_dir = resolve_tiff_directory(scan_dir)
    files = sorted(list(tiff_dir.glob("*.tif")) + list(tiff_dir.glob("*.tiff")))
    raw_window = patch_size * max(1, spatial_bin)
    if not preview_only and len(files) < 2 * raw_window:
        raise RuntimeError(
            f"[{state_name}] Need >= {2 * raw_window} slices in {tiff_dir}, found {len(files)}"
        )

    with tifffile.TiffFile(files[0]) as first_tif:
        shape_hw = tuple(first_tif.pages[0].shape)
        dtype = first_tif.pages[0].dtype

    roi_mask = make_circular_mask(shape_hw, radius_fraction=roi_radius_fraction)

    # Calibrate active-coating threshold if 3-phase is active
    delta_coating: float | None = None
    if eff_num_phases == 3:
        if delta_coating_threshold > delta_threshold:
            delta_coating = float(delta_coating_threshold)
        else:
            ref_indices = sorted({len(files) // 4, len(files) // 2, (3 * len(files)) // 4})
            ref_slices = [tifffile.imread(files[i]) for i in ref_indices]
            _, delta_coating = calibrate_multiphase_thresholds(
                ref_slices,
                roi_mask=roi_mask,
                sigma=sigma,
                fixed_delta_carbon=delta_threshold,
            )

    patch_dir = output_dir / f"{state_name}_binary_patches{patch_size}_{version}"
    qa_dir = patch_dir / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)

    mid_idx = len(files) // 2
    mid_raw = tifffile.imread(files[mid_idx])
    mid_delta, _, _ = compute_pore_referenced_delta(mid_raw, roi_mask=roi_mask, sigma=sigma)
    mid_seg, mid_stats = segment_slice_multiphase(
        mid_raw,
        delta_carbon=delta_threshold,
        delta_coating=delta_coating,
        roi_mask=roi_mask,
        sigma=sigma,
    )
    _save_multiphase_preview(
        out_path=qa_dir / "multiphase_overlay_diagnostic.png",
        state_name=state_name,
        material=resolved_material,
        raw_slice=mid_raw,
        seg_slice=mid_seg,
        delta_slice=mid_delta,
        roi_mask=roi_mask,
        delta_carbon=delta_threshold,
        delta_coating=delta_coating,
    )

    if preview_only:
        preview_info: dict[str, object] = {
            "state": state_name,
            "material": resolved_material,
            "beamline": calib.beamline,
            "load_mv_per_v": load_mv,
            "stress_kpa": stress_kpa,
            "effective_voxel_size_um": calib.effective_voxel_size_um,
            "num_phases": eff_num_phases,
            "delta_carbon": delta_threshold,
            "delta_coating": delta_coating,
            "mid_slice_stats": mid_stats,
            "preview_figure": str(qa_dir / "multiphase_overlay_diagnostic.png"),
        }
        with (qa_dir / "preview_summary.json").open("w") as f:
            json.dump(preview_info, f, indent=2)
        print(f"[{state_name}] Saved preview -> {qa_dir / 'multiphase_overlay_diagnostic.png'}", flush=True)
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
        raise RuntimeError(f"[{state_name}] No valid {raw_window}^2 XY positions inside ROI")

    print(
        f"[{state_name}] Pass 1/2: scanning {len(files)} slices "
        f"(material={resolved_material}, phases={eff_num_phases}, load={load_mv:.4f} mV/V)...",
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
            delta_carbon=delta_threshold,
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
            print(f"  [{state_name}] Scanned {idx}/{len(files)} slices", flush=True)

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
        state_name=state_name,
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

    print(f"[{state_name}] Pass 2/2: streaming clean 3D patches -> {patch_dir.name}", flush=True)
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
                        delta_carbon=delta_threshold,
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
            raise RuntimeError(f"[{state_name}] 0 {split_name} patches passed filter [{min_white}, {max_white}]")

        stacked = np.stack(accepted, axis=0)
        split_out = patch_dir / split_name
        split_out.mkdir(parents=True, exist_ok=True)
        np.save(split_out / "patches.npy", stacked)
        split_counts[split_name] = int(stacked.shape[0])
        split_sf_means[split_name] = float((stacked >= 1).mean())
        split_coat_means[split_name] = float((stacked == 2).mean())
        print(
            f"  [{state_name}] {split_name}: {stacked.shape[0]} patches "
            f"(solid={split_sf_means[split_name]:.4f}, coating={split_coat_means[split_name]:.4f})",
            flush=True,
        )

    sf_arr = np.asarray(solid_fractions, dtype=np.float64)
    cf_arr = np.asarray(coating_fractions, dtype=np.float64)
    clean_sf = sf_arr[valid_mask] if valid_mask.any() else sf_arr
    clean_cf = cf_arr[valid_mask] if valid_mask.any() else cf_arr
    clean_slices = int(valid_mask.sum())

    summary: dict[str, object] = {
        "state": state_name,
        "material": resolved_material,
        "beamline": calib.beamline,
        "source_folder": str(scan_dir),
        "patch_dir_name": patch_dir.name,
        "load_mv_per_v": load_mv,
        "stress_kpa": stress_kpa,
        "strain_z": calib.estimate_strain_z(clean_slices),
        "voxel_size_um": calib.voxel_size_um,
        "spatial_bin": spatial_bin,
        "effective_voxel_size_um": calib.effective_voxel_size_um,
        "num_phases": eff_num_phases,
        "delta_carbon": delta_threshold,
        "delta_coating": delta_coating,
        "is_repeat_or_secondary": _is_repeat_or_secondary(state_name),
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
                "dataset": state_name,
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


def build_bundle(
    drive_root: Path,
    output_dir: Path,
    datasets: list[str] | None = None,
    version: str = "carbon_calibrated_v1",
    material: str = "auto",
    num_phases: int = 2,
    auto_multiphase: bool = False,
    delta_threshold: float = 5500.0,
    delta_coating_threshold: float = 0.0,
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
    beamline: str = "primary",
    preview_only: bool = False,
    make_zip: bool = True,
) -> Path:
    scan_dirs = [drive_root / d for d in datasets] if datasets else discover_scan_folders(drive_root)
    if not scan_dirs:
        raise FileNotFoundError(f"No XCT scan folders discovered under {drive_root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    calib = CalibrationMetadata(
        voxel_size_um=voxel_size_um,
        spatial_bin=spatial_bin,
        kpa_per_mvv=kpa_per_mvv,
        beamline=beamline,
    )

    primary_states: dict[str, float] = {}
    repeat_states: dict[str, float] = {}
    by_material: dict[str, dict[str, float]] = {}
    scan_summaries: list[dict[str, object]] = []

    for scan_dir in scan_dirs:
        summary = stream_bundle_single_scan(
            scan_dir=scan_dir,
            output_dir=output_dir,
            version=version,
            material=material,
            num_phases=num_phases,
            auto_multiphase=auto_multiphase,
            delta_threshold=delta_threshold,
            delta_coating_threshold=delta_coating_threshold,
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
        )
        scan_summaries.append(summary)
        if preview_only:
            continue

        st_name = str(summary["state"])
        mat_name = str(summary["material"])
        mv = float(summary["load_mv_per_v"])  # type: ignore[arg-type]
        if summary["is_repeat_or_secondary"]:
            repeat_states[st_name] = mv
        else:
            primary_states[st_name] = mv
            by_material.setdefault(mat_name, {})[st_name] = mv

    if preview_only:
        preview_manifest = output_dir / "preview_manifest.json"
        with preview_manifest.open("w") as f:
            json.dump({"previews": scan_summaries}, f, indent=2)
        return preview_manifest

    with (output_dir / "primary_states.json").open("w") as f:
        json.dump(dict(sorted(primary_states.items(), key=lambda kv: kv[1])), f, indent=2)

    manifest = {
        "version": version,
        "patch_size": patch_size,
        "spatial_bin": spatial_bin,
        "voxel_size_um": voxel_size_um,
        "effective_voxel_size_um": calib.effective_voxel_size_um,
        "kpa_per_mvv": kpa_per_mvv,
        "beamline": beamline,
        "delta_threshold": delta_threshold,
        "primary_states": dict(sorted(primary_states.items(), key=lambda kv: kv[1])),
        "states_by_material": {
            m: dict(sorted(d.items(), key=lambda kv: kv[1])) for m, d in sorted(by_material.items())
        },
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
    if len(sys.argv) == 1 and SPYDER_DRIVE_ROOT.exists():
        build_bundle(
            drive_root=SPYDER_DRIVE_ROOT,
            output_dir=SPYDER_OUTPUT_DIR,
            auto_multiphase=True,
            spatial_bin=2,
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
    parser.add_argument("--delta-threshold", type=float, default=5500.0)
    parser.add_argument(
        "--delta-coating-threshold",
        type=float,
        default=0.0,
        help="Explicit coating delta threshold (> delta-threshold); 0 auto-calibrates via 2-stage solid Otsu",
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
    parser.add_argument("--beamline", default="primary")
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Fast 10s check: save multiphase_overlay_diagnostic.png without extracting 3D patches",
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
        delta_threshold=args.delta_threshold,
        delta_coating_threshold=args.delta_coating_threshold,
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
        beamline=args.beamline,
        preview_only=args.preview_only,
        make_zip=args.zip,
    )


if __name__ == "__main__":
    main()
