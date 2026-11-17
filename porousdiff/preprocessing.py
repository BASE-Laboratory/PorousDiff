"""Physics-consistent XCT segmentation, artifact gating, and stratified 3D patch extraction."""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.ndimage import binary_closing, binary_dilation, gaussian_filter
from skimage.filters import threshold_otsu


@dataclass(frozen=True)
class CalibrationMetadata:
    """Physical calibration linking beamline XCT coordinates and load-cell readings."""

    voxel_size_um: float = 1.0
    spatial_bin: int = 1
    kpa_per_mvv: float | None = None
    zero_offset_mvv: float = 0.0
    reference_thickness_slices: int | None = None
    beamline: str = "primary"

    @property
    def effective_voxel_size_um(self) -> float:
        return float(self.voxel_size_um * max(1, self.spatial_bin))

    def load_to_stress_kpa(self, load_mv_v: float) -> float | None:
        if self.kpa_per_mvv is None:
            return None
        return float(max(0.0, (load_mv_v - self.zero_offset_mvv) * self.kpa_per_mvv))

    def estimate_strain_z(self, clean_thickness_slices: int) -> float | None:
        if self.reference_thickness_slices is None or self.reference_thickness_slices <= 0:
            return None
        ratio = float(clean_thickness_slices) / float(self.reference_thickness_slices)
        return float(np.clip(1.0 - ratio, 0.0, 0.95))


def infer_material_name(name: str) -> str:
    """Infer canonical electrode material key from dataset/folder name."""
    low = name.lower()
    if "tio2" in low or "anode" in low:
        return "tio2_anode"
    if "cuhcf" in low or "cuchf" in low or "hexacyano" in low or "cathode" in low:
        return "cuhcf_cathode"
    return "virgin_carbon"


def parse_load_value(name: str) -> float:
    """Parse load-cell reading in mV/V from folder/state names.

    Handles both Diamond synchrotron naming (`crbon_virgin_1_0p048mVV_`, `0p008mVV`)
    and Isambard shorthand (`Op008mVV` -> `0.008`, `Op033mVV` -> `0.033`).
    """
    match = re.search(r"(?:0|[Oo])p(\d+)mVV", name)
    if match:
        digits = match.group(1)
        return float(f"0.{digits}")

    match_num = re.search(r"(\d+(?:\.\d+)?)\s*mV", name, flags=re.IGNORECASE)
    if match_num:
        val = float(match_num.group(1))
        return val / 1000.0 if val >= 1.0 else val

    raise ValueError(
        f"Cannot parse load value from {name!r}; supply an explicit states mapping."
    )


def resolve_tiff_directory(dataset_path: Path) -> Path:
    """Locate `.tif` files directly in `dataset_path` or inside `rec_16bit_Paganin`."""
    direct = list(dataset_path.glob("*.tif")) + list(dataset_path.glob("*.tiff"))
    if direct:
        return dataset_path
    paganin = dataset_path / "rec_16bit_Paganin"
    if paganin.is_dir():
        sub = list(paganin.glob("*.tif")) + list(paganin.glob("*.tiff"))
        if sub:
            return paganin
    raise FileNotFoundError(f"No TIFF slices found under {dataset_path}")


def make_circular_mask(
    shape_hw: tuple[int, int],
    radius_fraction: float = 0.70,
    center: tuple[float, float] | None = None,
) -> np.ndarray:
    """Boolean mask for the inscribed circular electrode punch inside a square XCT slice."""
    if not 0.0 < radius_fraction <= 1.0:
        raise ValueError("radius_fraction must be in (0, 1]")
    h, w = shape_hw
    yc, xc = center if center is not None else (h / 2.0, w / 2.0)
    radius = 0.5 * min(h, w) * radius_fraction
    yy, xx = np.ogrid[:h, :w]
    return ((yy - yc) ** 2 + (xx - xc) ** 2) <= (radius**2)


def estimate_pore_mode(values: np.ndarray) -> tuple[float, float]:
    """Estimate the dominant pore/air intensity mode and robust spread (MAD).

    In porous electrodes (porosity > 50%), the lower/primary histogram mode inside
    the sample ROI corresponds to the pore phase.
    """
    flat = np.asarray(values, dtype=np.float32).ravel()
    if flat.size == 0:
        raise ValueError("Cannot estimate pore mode on empty array")

    lo, hi = np.percentile(flat, [0.5, 90.0])
    if hi <= lo:
        med = float(np.median(flat))
        return med, 1.0

    # Restrict search for the pore peak to below the 75th percentile so dense solid
    # phases never hijack the mode even under heavy compression.
    upper_search = float(np.percentile(flat, 75.0))
    sub = flat[(flat >= lo) & (flat <= max(upper_search, lo + 1.0))]
    if sub.size < 32:
        sub = flat

    hist, edges = np.histogram(sub, bins=128)
    smooth_hist = np.convolve(hist, [0.25, 0.5, 0.25], mode="same")
    peak_idx = int(np.argmax(smooth_hist))
    pore_mode = 0.5 * float(edges[peak_idx] + edges[peak_idx + 1])

    pore_side = flat[flat <= pore_mode + (pore_mode - float(np.percentile(flat, 5.0)))]
    if pore_side.size < 16:
        pore_side = sub
    mad = float(np.median(np.abs(pore_side - pore_mode))) * 1.4826
    return pore_mode, max(mad, 1.0)


def compute_pore_referenced_delta(
    image: np.ndarray,
    roi_mask: np.ndarray | None = None,
    sigma: float = 32.0,
) -> tuple[np.ndarray, float, float]:
    """Subtract gentle low-frequency pore background without hollowing out dense fibers.

    Clamping bright fiber pixels near `pore_mode + 2 * pore_std` before Gaussian
    smoothing removes slice-to-slice synchrotron beam flicker and Paganin radial
    cupping while keeping high fiber-packing regions intact under compression.
    """
    img = np.asarray(image, dtype=np.float32)
    roi_vals = img[roi_mask] if roi_mask is not None else img
    pore_mode, pore_std = estimate_pore_mode(roi_vals)

    if sigma > 0:
        clamped = np.minimum(img, pore_mode + 2.0 * pore_std)
        if roi_mask is not None:
            clamped = np.where(roi_mask, clamped, pore_mode)
        background = gaussian_filter(clamped, sigma=sigma)
    else:
        background = np.full_like(img, pore_mode)

    delta = img - background
    return delta, pore_mode, pore_std


def calibrate_delta_threshold(
    reference_slices: Iterable[np.ndarray],
    roi_mask: np.ndarray | None = None,
    sigma: float = 32.0,
) -> float:
    """Calibrate a single shared `delta_threshold` from reference XCT slices."""
    pooled = []
    rng = np.random.default_rng(0)
    for slc in reference_slices:
        delta, _, _ = compute_pore_referenced_delta(slc, roi_mask=roi_mask, sigma=sigma)
        vals = delta[roi_mask] if roi_mask is not None else delta.ravel()
        if vals.size > 50_000:
            vals = rng.choice(vals, size=50_000, replace=False)
        pooled.append(vals)

    if not pooled:
        raise ValueError("No reference slices provided for threshold calibration")

    all_vals = np.concatenate(pooled)
    lo, hi = np.percentile(all_vals, [1.0, 99.8])
    clipped = all_vals[(all_vals >= lo) & (all_vals <= hi)]
    return float(threshold_otsu(clipped))


def _plateau_isodata_threshold(vals: np.ndarray, nbins: int = 512) -> float:
    """Compute a gap-centered Otsu threshold refined to the half-maximum class midpoint.

    Standard `skimage.filters.threshold_otsu` selects the *leftmost* histogram bin on
    empty intensity gaps and shifts toward the dominant mode under 90%+ class imbalance.
    Centering the optimal Otsu plateau and refining via 1D IsoData yields the unbiased
    50% partial-volume iso-contour `0.5 * (mu_low + mu_high)` for Paganin XCT interfaces.
    """
    hist, bin_edges = np.histogram(vals, bins=nbins)
    hist_f = hist.astype(np.float64)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    w0 = np.cumsum(hist_f)
    w1 = w0[-1] - w0
    valid = (w0 > 0) & (w1 > 0)
    if not np.any(valid):
        return float(np.median(vals))

    csum = np.cumsum(hist_f * centers)
    m0 = csum[valid] / w0[valid]
    m1 = (csum[-1] - csum[valid]) / w1[valid]
    var_between = w0[valid] * w1[valid] * (m0 - m1) ** 2
    best = var_between >= 0.999 * var_between.max()
    best_centers = centers[valid][best]
    thresh = float(0.5 * (best_centers[0] + best_centers[-1]))

    for _ in range(3):
        lo = vals[vals <= thresh]
        hi = vals[vals > thresh]
        if lo.size == 0 or hi.size == 0:
            break
        thresh = float(0.5 * (lo.mean() + hi.mean()))
    return thresh


def calibrate_multiphase_thresholds(
    reference_slices: Iterable[np.ndarray],
    roi_mask: np.ndarray | None = None,
    sigma: float = 32.0,
    fixed_delta_carbon: float | None = None,
) -> tuple[float, float]:
    """Calibrate `(delta_carbon, delta_coating)` thresholds for 3-phase coated electrodes.

    Gating above `4 * pore_std` isolates solid voxels (`carbon_fiber + active_coating`)
    from the 80–95% pore background so `delta_coating` cleanly splits carbon (`Z=6`) vs
    high-Z coating (`TiO2`/`CuHCF`), and `delta_carbon` lands on the unbiased 50%
    partial-volume midpoint between pore and carbon without high-Z skew.
    """
    pooled: list[np.ndarray] = []
    pore_stds: list[float] = []
    rng = np.random.default_rng(0)
    for slc in reference_slices:
        delta, _, p_std = compute_pore_referenced_delta(slc, roi_mask=roi_mask, sigma=sigma)
        pore_stds.append(p_std)
        vals = delta[roi_mask] if roi_mask is not None else delta.ravel()
        if vals.size > 80_000:
            vals = rng.choice(vals, size=80_000, replace=False)
        pooled.append(vals)

    if not pooled:
        raise ValueError("No reference slices provided for multiphase calibration")

    all_vals = np.concatenate(pooled)
    med_pstd = max(float(np.median(pore_stds)), 1.0)
    solid_gate = 4.0 * med_pstd

    if fixed_delta_carbon is not None:
        delta_carbon = float(fixed_delta_carbon)
        solid_candidates = all_vals[all_vals > delta_carbon]
    else:
        solid_candidates = all_vals[all_vals > solid_gate]

    if solid_candidates.size < 32:
        d_carb = float(fixed_delta_carbon) if fixed_delta_carbon is not None else max(solid_gate, 1000.0)
        return d_carb, d_carb + max(4.0 * med_pstd, 2000.0)

    hi_solid = float(np.percentile(solid_candidates, 99.9))
    solid_clipped = solid_candidates[solid_candidates <= hi_solid]
    if solid_clipped.size < 32 or float(solid_clipped.max() - solid_clipped.min()) < 1e-3:
        d_carb = float(fixed_delta_carbon) if fixed_delta_carbon is not None else max(solid_gate, 1000.0)
        return d_carb, d_carb + max(4.0 * med_pstd, 2000.0)

    delta_coating = _plateau_isodata_threshold(solid_clipped)

    if fixed_delta_carbon is None:
        pore_vals = all_vals[all_vals <= solid_gate]
        carbon_vals = solid_clipped[solid_clipped <= delta_coating]
        if pore_vals.size > 0 and carbon_vals.size > 0:
            delta_carbon = max(
                0.5 * (float(pore_vals.mean()) + float(carbon_vals.mean())),
                3.5 * med_pstd,
            )
        else:
            delta_carbon = max(_plateau_isodata_threshold(all_vals), 3.5 * med_pstd)

    delta_coating = max(float(delta_coating), float(delta_carbon) + max(2.0 * med_pstd, 500.0))
    return float(delta_carbon), float(delta_coating)


def segment_slice_calibrated(
    image: np.ndarray,
    delta_threshold: float = 5500.0,
    roi_mask: np.ndarray | None = None,
    sigma: float = 32.0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Segment a 2D XCT slice using a consistent pore-referenced physical threshold."""
    if delta_threshold <= 0:
        raise ValueError("delta_threshold must be positive")

    delta, pore_mode, pore_std = compute_pore_referenced_delta(
        image, roi_mask=roi_mask, sigma=sigma
    )
    binary = (delta > delta_threshold).astype(np.uint8)
    if roi_mask is not None:
        binary = np.where(roi_mask, binary, 0).astype(np.uint8)
        solid_fraction = float(binary[roi_mask].mean())
    else:
        solid_fraction = float(binary.mean())

    stats = {
        "pore_mode": pore_mode,
        "pore_std": pore_std,
        "solid_fraction": solid_fraction,
    }
    return binary, stats


def segment_slice_multiphase(
    image: np.ndarray,
    delta_carbon: float = 5500.0,
    delta_coating: float | None = None,
    roi_mask: np.ndarray | None = None,
    sigma: float = 32.0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Segment a 2D XCT slice into `{0: pore, 1: carbon_fiber, 2: active_coating}`.

    When `delta_coating is None`, behaves identically to 2-phase `segment_slice_calibrated`.
    """
    if delta_carbon <= 0:
        raise ValueError("delta_carbon must be positive")
    if delta_coating is not None and delta_coating <= delta_carbon:
        raise ValueError("delta_coating must be strictly greater than delta_carbon")

    delta, pore_mode, pore_std = compute_pore_referenced_delta(
        image, roi_mask=roi_mask, sigma=sigma
    )
    labels = np.zeros(image.shape, dtype=np.uint8)
    labels[delta > delta_carbon] = 1
    if delta_coating is not None:
        labels[delta > delta_coating] = 2

    if roi_mask is not None:
        labels = np.where(roi_mask, labels, 0).astype(np.uint8)
        region = labels[roi_mask]
    else:
        region = labels.ravel()

    solid_frac = float((region >= 1).mean())
    carbon_frac = float((region == 1).mean())
    coating_frac = float((region == 2).mean())
    stats = {
        "pore_mode": pore_mode,
        "pore_std": pore_std,
        "solid_fraction": solid_frac,
        "carbon_fraction": carbon_frac,
        "coating_fraction": coating_frac,
    }
    return labels, stats


def downsample_volume_rev(volume: np.ndarray, factor: int = 2) -> np.ndarray:
    """Topology- and volume-fraction-preserving 3D block binning for REV-scale patches.

    Pools `(factor x factor x factor)` voxel blocks using block occupancy ranked with
    connected-neighborhood tie-breaking so global solid and active-coating volume
    fractions are conserved (`< 0.5%` drift) without eroding thin carbon fibers.
    """
    if factor < 1:
        raise ValueError("factor must be >= 1")
    vol = np.asarray(volume, dtype=np.uint8)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {vol.shape}")
    if factor == 1:
        return vol.copy()

    d, h, w = vol.shape
    od, oh, ow = d // factor, h // factor, w // factor
    if min(od, oh, ow) < 1:
        raise ValueError(f"Volume shape {vol.shape} smaller than bin factor {factor}")

    cropped = vol[: od * factor, : oh * factor, : ow * factor]
    blocks = cropped.reshape(od, factor, oh, factor, ow, factor)

    solid_mask = cropped >= 1
    target_solid_frac = float(solid_mask.mean())
    n_out = od * oh * ow
    target_solid_count = int(round(target_solid_frac * n_out))

    out = np.zeros((od, oh, ow), dtype=np.uint8)
    if target_solid_count == 0:
        return out

    occ_solid = (blocks >= 1).mean(axis=(1, 3, 5)).astype(np.float32)
    # Break ties along continuous 3D fiber backbones while never activating empty blocks
    score_solid = np.where(
        occ_solid > 0.0,
        occ_solid + 1e-2 * gaussian_filter(occ_solid, sigma=0.8),
        0.0,
    )
    nonzero_solid = int(np.count_nonzero(score_solid > 0.0))
    k_solid = min(target_solid_count, nonzero_solid)
    if k_solid > 0:
        flat_scores = score_solid.ravel()
        top_idx = np.argpartition(flat_scores, -k_solid)[-k_solid:]
        out.ravel()[top_idx] = 1

    if np.any(cropped == 2):
        target_coat_frac = float((cropped == 2).mean())
        target_coat_count = int(round(target_coat_frac * n_out))
        if target_coat_count > 0:
            occ_coat = (blocks == 2).mean(axis=(1, 3, 5)).astype(np.float32)
            score_coat = np.where(
                (out == 1) & (occ_coat > 0.0),
                occ_coat + 1e-2 * gaussian_filter(occ_coat, sigma=0.8),
                0.0,
            )
            nonzero_coat = int(np.count_nonzero(score_coat > 0.0))
            k_coat = min(target_coat_count, nonzero_coat)
            if k_coat > 0:
                flat_coat = score_coat.ravel()
                coat_idx = np.argpartition(flat_coat, -k_coat)[-k_coat:]
                out.ravel()[coat_idx] = 2

    return out


def compute_radial_ring_score(
    image: np.ndarray,
    max_radius_fraction: float = 0.40,
    center: tuple[float, float] | None = None,
) -> float:
    """Scale-invariant concentric ring score via split-half detrended radial covariance.

    Splits the slice into disjoint upper and lower semicircles around `center`,
    removes linear radial cupping, and measures their normalized radial covariance.
    Uncorrelated fibers in opposite halves have zero expected covariance across both
    small crops (`96x96`) and full `2560x2560` XCT slices (`0.00 - 0.08`), whereas
    concentric bullseye ring seams and plunger interference bands spike (`> 0.15`).
    """
    img = np.asarray(image, dtype=np.float32)
    h, w = img.shape
    yc, xc = center if center is not None else (h / 2.0, w / 2.0)
    r_max = int(0.5 * min(h, w) * max_radius_fraction)
    if r_max < 12:
        return 0.0

    # Subsample large slices by 2x for speed without losing ring structure
    step = 2 if min(h, w) >= 512 else 1
    sub = img[::step, ::step]
    sh, sw = sub.shape
    syc, sxc = yc / step, xc / step
    sr_max = max(12, r_max // step)

    yy, xx = np.indices((sh, sw))
    rr = np.sqrt((yy - syc) ** 2 + (xx - sxc) ** 2).astype(np.int32)
    mask = (rr >= 4) & (rr < sr_max)
    total_var = float(np.var(sub[mask]))
    if total_var < 1e-6:
        return 0.0

    top = mask & (yy < syc - 1.5)
    bot = mask & (yy > syc + 1.5)
    s_top = np.bincount(rr[top], weights=sub[top], minlength=sr_max)
    c_top = np.bincount(rr[top], minlength=sr_max)
    s_bot = np.bincount(rr[bot], weights=sub[bot], minlength=sr_max)
    c_bot = np.bincount(rr[bot], minlength=sr_max)

    valid = (c_top >= 6) & (c_bot >= 6)
    if valid.sum() < 6:
        return 0.0

    r_val = np.nonzero(valid)[0].astype(np.float32)
    r_val -= r_val.mean()
    denom = max(float(np.dot(r_val, r_val)), 1e-6)

    m_top = s_top[valid] / c_top[valid]
    m_bot = s_bot[valid] / c_bot[valid]
    m_top = m_top - m_top.mean() - (float(np.dot(r_val, m_top)) / denom) * r_val
    m_bot = m_bot - m_bot.mean() - (float(np.dot(r_val, m_bot)) / denom) * r_val

    cov = float(np.mean(m_top * m_bot))
    return float(np.clip(cov / total_var, 0.0, 1.0))


def detect_valid_z_intervals(
    ring_scores: Sequence[float],
    pore_modes: Sequence[float],
    slice_jumps: Sequence[float],
    min_interval_length: int = 96,
    max_ring_score: float = 0.15,
    jump_factor: float = 3.5,
    max_pore_mode_drift: float = 6500.0,
    seam_margin: int = 12,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Identify artifact-free Z intervals excluding bullseye seams and rig plungers.

    Returns:
        valid_mask: boolean array of shape `(num_slices,)`
        intervals: list of `(z_start, z_end)` half-open clean intervals with length >= `min_interval_length`
    """
    rings = np.asarray(ring_scores, dtype=np.float32)
    modes = np.asarray(pore_modes, dtype=np.float32)
    jumps = np.asarray(slice_jumps, dtype=np.float32)
    n = rings.size
    if not (modes.size == n and jumps.size == n):
        raise ValueError("Slice quality arrays must have equal length")

    clean_seed = rings <= max_ring_score
    ref_modes = modes[clean_seed] if clean_seed.any() else modes
    ref_jumps = jumps[clean_seed] if clean_seed.any() else jumps

    med_mode = float(np.median(ref_modes))
    med_jump = max(float(np.median(ref_jumps)), 1e-6)

    # Solid plungers/seams brighten the mode (> +max_pore_mode_drift), whereas Paganin
    # dark halos around dense compressed fibers can dip locally without being artifacts.
    bad = (
        (rings > max_ring_score)
        | (jumps > jump_factor * med_jump)
        | (modes > med_mode + max_pore_mode_drift)
        | (modes < med_mode - 2.5 * max_pore_mode_drift)
    )

    # Close narrow gaps inside multi-slice seam/plunger zones, then pad by `seam_margin`
    if bad.any():
        close_width = max(3, seam_margin)
        bad = binary_closing(bad, structure=np.ones(close_width, dtype=bool))
        if seam_margin > 0:
            pad_struct = np.ones(2 * seam_margin + 1, dtype=bool)
            bad = binary_dilation(bad, structure=pad_struct)

    valid_mask = ~bad
    intervals: list[tuple[int, int]] = []
    start: int | None = None
    for z, ok in enumerate(valid_mask):
        if ok and start is None:
            start = z
        elif not ok and start is not None:
            if z - start >= min_interval_length:
                intervals.append((start, z))
            start = None
    if start is not None and n - start >= min_interval_length:
        intervals.append((start, n))

    # Zero out valid_mask in fragments shorter than min_interval_length
    filtered_mask = np.zeros(n, dtype=bool)
    for s, e in intervals:
        filtered_mask[s:e] = True

    return filtered_mask, intervals


def split_z_intervals_with_buffer(
    intervals: Sequence[tuple[int, int]],
    patch_size: int,
    val_fraction: float = 0.2,
    buffer_slices: int | None = None,
) -> dict[str, list[tuple[int, int]]]:
    """Split clean Z intervals into spatially disjoint `train` and `val` slabs.

    Inserts an unused buffer of at least `patch_size` slices wherever a contiguous
    clean interval is partitioned between `train` and `val`, ensuring zero shared or
    immediately adjacent voxels across splits.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    buf = patch_size if buffer_slices is None else max(0, buffer_slices)

    usable = [(s, e) for s, e in intervals if (e - s) >= patch_size]
    if not usable:
        raise RuntimeError(
            f"No clean Z intervals of length >= {patch_size} available"
        )

    total_clean = sum(e - s for s, e in usable)
    target_val = max(patch_size, int(round(total_clean * val_fraction)))

    # Carve validation from the longest clean interval (or trailing clean interval if large enough)
    train_intervals: list[tuple[int, int]] = []
    val_intervals: list[tuple[int, int]] = []

    # Check if the last interval alone can serve as validation without splitting
    if len(usable) >= 2 and usable[-1][1] - usable[-1][0] >= patch_size:
        last_s, last_e = usable[-1]
        last_len = last_e - last_s
        if last_len <= max(target_val * 2, patch_size * 2):
            train_intervals.extend(usable[:-1])
            val_intervals.append((last_s, last_e))
            return {"train": train_intervals, "val": val_intervals}

    # Otherwise split the longest interval with an explicit buffer gap
    longest_idx = int(np.argmax([e - s for s, e in usable]))
    for idx, (s, e) in enumerate(usable):
        if idx != longest_idx:
            train_intervals.append((s, e))
            continue

        span = e - s
        val_len = max(patch_size, min(target_val, span - patch_size - buf))
        train_end = e - val_len - buf
        val_start = e - val_len
        if train_end - s < patch_size or e - val_start < patch_size:
            # Fallback if interval is tight: shrink buffer only if unavoidable, else error
            if span >= 2 * patch_size:
                mid = s + (span - patch_size) // 2
                train_end = max(s + patch_size, e - patch_size - min(buf, span - 2 * patch_size))
                val_start = e - patch_size
            else:
                raise RuntimeError(
                    f"Clean Z span ({span} slices) cannot fit disjoint train and val patches "
                    f"of size {patch_size} (need >= {2 * patch_size + buf} slices)."
                )
        train_intervals.append((s, train_end))
        val_intervals.append((val_start, e))

    return {"train": train_intervals, "val": val_intervals}


def enumerate_cylinder_xy_positions(
    shape_hw: tuple[int, int],
    patch_size: int,
    stride_xy: int,
    radius_fraction: float = 0.70,
    center: tuple[float, float] | None = None,
) -> list[tuple[int, int]]:
    """Return `(y, x)` top-left patch coordinates whose entire square lies inside the circular ROI."""
    h, w = shape_hw
    if h < patch_size or w < patch_size:
        raise ValueError(f"Slice shape {shape_hw} smaller than patch_size={patch_size}")

    yc, xc = center if center is not None else (h / 2.0, w / 2.0)
    radius_sq = (0.5 * min(h, w) * radius_fraction) ** 2

    positions: list[tuple[int, int]] = []
    for y in range(0, h - patch_size + 1, stride_xy):
        for x in range(0, w - patch_size + 1, stride_xy):
            corners = (
                (y, x),
                (y, x + patch_size - 1),
                (y + patch_size - 1, x),
                (y + patch_size - 1, x + patch_size - 1),
            )
            if all((cy - yc) ** 2 + (cx - xc) ** 2 <= radius_sq for cy, cx in corners):
                positions.append((y, x))

    return positions


def build_stratified_patch_plan(
    z_intervals: Sequence[tuple[int, int]],
    xy_positions: Sequence[tuple[int, int]],
    patch_size: int,
    stride_z: int,
    max_patches: int,
    rng: np.random.Generator,
) -> list[tuple[int, list[tuple[int, int]]]]:
    """Distribute patch budget uniformly across all valid Z-windows (round-robin).

    Prevents exhausting `max_patches` inside the first 2-3 Z-windows and guarantees
    the full clean thickness of the electrode is represented.
    """
    if not xy_positions:
        raise ValueError("No valid XY positions inside circular ROI")

    z_windows: list[int] = []
    for s, e in z_intervals:
        win = list(range(s, e - patch_size + 1, stride_z))
        # Ensure the trailing edge of the interval can also be sampled if missed by stride_z
        last_valid = e - patch_size
        if last_valid >= s and (not win or win[-1] != last_valid):
            if not win or (last_valid - win[-1]) >= max(1, stride_z // 2):
                win.append(last_valid)
        z_windows.extend(win)

    if not z_windows:
        raise ValueError("No valid Z windows available")

    z_order = list(z_windows)
    rng.shuffle(z_order)

    per_z_queues: dict[int, list[tuple[int, int]]] = {}
    for z in z_order:
        xy_copy = list(xy_positions)
        rng.shuffle(xy_copy)
        per_z_queues[z] = xy_copy

    # Allocate extra candidate headroom (1.5x max_patches) in round-robin order
    # so min/max solid-fraction filters still maintain uniform Z coverage.
    target_candidates = min(
        len(z_order) * len(xy_positions),
        max(max_patches, int(max_patches * 1.5) + len(z_order)),
    )

    selected_per_z: dict[int, list[tuple[int, int]]] = {z: [] for z in z_order}
    total_planned = 0
    cursor = 0
    while total_planned < target_candidates:
        progressed = False
        for z in z_order:
            if cursor < len(per_z_queues[z]):
                selected_per_z[z].append(per_z_queues[z][cursor])
                total_planned += 1
                progressed = True
                if total_planned >= target_candidates:
                    break
        if not progressed:
            break
        cursor += 1

    # Sort Z ascending for sequential disk reads during extraction
    return [(z, selected_per_z[z]) for z in sorted(selected_per_z) if selected_per_z[z]]
