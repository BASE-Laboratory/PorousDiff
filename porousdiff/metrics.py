"""3D physical and morphological validation metrics for porous electrode volumes."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import tifffile
from scipy.ndimage import label


def validate_binary_volume(volume: np.ndarray) -> np.ndarray:
    """Validate and cast a 3D `{0, 1}` microstructure volume `(Z, Y, X)`."""
    vol = np.asarray(volume)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume (Z, Y, X), got shape {vol.shape}")
    return (vol > 0).astype(np.uint8)


def solid_fraction(volume: np.ndarray) -> float:
    vol = validate_binary_volume(volume)
    return float(vol.mean())


def specific_surface_area(volume: np.ndarray, voxel_size: float = 1.0) -> dict[str, float]:
    """Compute exposed pore-solid interfacial area per unit volume (`1 / voxel_size`)."""
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    vol = validate_binary_volume(volume)
    n_voxels = float(vol.size)

    faces_z = float(np.count_nonzero(vol[1:, :, :] != vol[:-1, :, :]))
    faces_y = float(np.count_nonzero(vol[:, 1:, :] != vol[:, :-1, :]))
    faces_x = float(np.count_nonzero(vol[:, :, 1:] != vol[:, :, :-1]))

    total_faces = faces_z + faces_y + faces_x
    scale = n_voxels * voxel_size
    return {
        "ssa_total": total_faces / scale,
        "ssa_z": faces_z / scale,
        "ssa_y": faces_y / scale,
        "ssa_x": faces_x / scale,
    }


def _chord_lengths_1d(lines_2d: np.ndarray, phase_val: int) -> np.ndarray:
    """Vectorized run-length chord extraction along the last axis of a 2D array `(N_lines, L)`."""
    mask = (lines_2d == phase_val).astype(np.int8)
    n_lines, length = mask.shape
    if length == 0:
        return np.empty(0, dtype=np.float32)

    padded = np.zeros((n_lines, length + 2), dtype=np.int8)
    padded[:, 1:-1] = mask
    diffs = np.diff(padded, axis=1)

    starts_r, starts_c = np.nonzero(diffs == 1)
    ends_r, ends_c = np.nonzero(diffs == -1)
    if starts_c.size == 0:
        return np.empty(0, dtype=np.float32)

    # Because rows are sorted identically in `np.nonzero`, starts and ends pair 1-to-1
    assert np.array_equal(starts_r, ends_r)
    chords = (ends_c - starts_c).astype(np.float32)

    # Prefer strictly interior (uncensored) chords if enough exist, else keep all chords
    interior = (starts_c > 0) & (ends_c < length)
    if np.count_nonzero(interior) >= 16:
        return chords[interior]
    return chords


def directional_chord_metrics(volume: np.ndarray, phase: int = 0) -> dict[str, float]:
    """Mean chord lengths along Z (through-plane) and X, Y (in-plane) + anisotropy."""
    vol = validate_binary_volume(volume)
    if phase not in (0, 1):
        raise ValueError("phase must be 0 (pore) or 1 (solid)")

    # Transpose target axis to last dimension and flatten remaining 2 dims
    z_lines = np.moveaxis(vol, 0, -1).reshape(-1, vol.shape[0])
    y_lines = np.moveaxis(vol, 1, -1).reshape(-1, vol.shape[1])
    x_lines = vol.reshape(-1, vol.shape[2])

    cz = _chord_lengths_1d(z_lines, phase)
    cy = _chord_lengths_1d(y_lines, phase)
    cx = _chord_lengths_1d(x_lines, phase)

    mean_z = float(cz.mean()) if cz.size else 0.0
    mean_y = float(cy.mean()) if cy.size else 0.0
    mean_x = float(cx.mean()) if cx.size else 0.0
    inplane = 0.5 * (mean_x + mean_y)
    anisotropy = float(inplane / mean_z) if mean_z > 1e-6 else 0.0

    prefix = "pore" if phase == 0 else "solid"
    return {
        f"{prefix}_chord_z": mean_z,
        f"{prefix}_chord_y": mean_y,
        f"{prefix}_chord_x": mean_x,
        f"{prefix}_anisotropy_xy_over_z": anisotropy,
    }


def directional_two_point_correlation(
    volume: np.ndarray,
    max_lag: int | None = None,
) -> dict[str, np.ndarray]:
    """Exact non-periodic two-point correlation `S_2(r)` along `Z`, `Y`, and `X`."""
    vol = validate_binary_volume(volume).astype(np.float64)
    d, h, w = vol.shape
    limit = min(d, h, w) // 2 if max_lag is None else min(max_lag, d - 1, h - 1, w - 1)

    curves: dict[str, np.ndarray] = {"lags": np.arange(limit + 1, dtype=np.int32)}
    for axis, name in ((0, "s2_z"), (1, "s2_y"), (2, "s2_x")):
        n = vol.shape[axis]
        pad_len = 1 << (2 * n - 1).bit_length()
        spec = np.fft.rfft(vol, n=pad_len, axis=axis)
        ac = np.fft.irfft(spec * np.conj(spec), n=pad_len, axis=axis)

        slc = [slice(None)] * 3
        slc[axis] = slice(0, limit + 1)
        summed = ac[tuple(slc)].sum(axis=tuple(i for i in range(3) if i != axis))
        perp_count = float(vol.size // n)
        valid_pairs = (n - np.arange(limit + 1, dtype=np.float64)) * perp_count
        curves[name] = (summed / np.maximum(valid_pairs, 1.0)).astype(np.float32)

    return curves


def connectivity_and_percolation(volume: np.ndarray) -> dict[str, float]:
    """3D 6-connected solid network integrity and directional pore percolation."""
    vol = validate_binary_volume(volume)
    struct6 = np.zeros((3, 3, 3), dtype=np.uint8)
    struct6[1, 1, :] = 1
    struct6[1, :, 1] = 1
    struct6[:, 1, 1] = 1

    # Solid network connectivity
    total_solid = int(vol.sum())
    if total_solid == 0:
        largest_solid_frac = 0.0
    else:
        solid_labels, n_solid = label(vol == 1, structure=struct6)
        if n_solid == 0:
            largest_solid_frac = 0.0
        else:
            counts = np.bincount(solid_labels.ravel())[1:]
            largest_solid_frac = float(counts.max() / total_solid)

    # Pore percolation across opposite faces
    pore_labels, _ = label(vol == 0, structure=struct6)

    def _spans(lbl_a: np.ndarray, lbl_b: np.ndarray) -> float:
        a = set(np.unique(lbl_a)) - {0}
        b = set(np.unique(lbl_b)) - {0}
        return 1.0 if (a & b) else 0.0

    return {
        "solid_connected_fraction": largest_solid_frac,
        "pore_percolates_z": _spans(pore_labels[0, :, :], pore_labels[-1, :, :]),
        "pore_percolates_y": _spans(pore_labels[:, 0, :], pore_labels[:, -1, :]),
        "pore_percolates_x": _spans(pore_labels[:, :, 0], pore_labels[:, :, -1]),
    }


def geodesic_pore_tortuosity(volume: np.ndarray, axis: int = 0) -> float:
    """Fast vectorized 6-connected shortest-path geodesic tortuosity across `axis`.

    Returns `nan` if the pore space does not percolate from inlet (`0`) to outlet (`L - 1`).
    """
    vol = validate_binary_volume(volume)
    pore = np.moveaxis(vol == 0, axis, 0)
    depth = pore.shape[0]
    if depth < 2:
        return 1.0

    dist = np.full(pore.shape, -1, dtype=np.int32)
    frontier = np.zeros_like(pore, dtype=bool)
    frontier[0] = pore[0]
    if not frontier[0].any():
        return float("nan")

    dist[frontier] = 0
    unvisited = pore & (~frontier)

    step = 0
    max_steps = pore.size
    while step < max_steps:
        step += 1
        nxt = np.zeros_like(frontier)
        nxt[1:, :, :] |= frontier[:-1, :, :]
        nxt[:-1, :, :] |= frontier[1:, :, :]
        nxt[:, 1:, :] |= frontier[:, :-1, :]
        nxt[:, :-1, :] |= frontier[:, 1:, :]
        nxt[:, :, 1:] |= frontier[:, :, :-1]
        nxt[:, :, :-1] |= frontier[:, :, 1:]
        nxt &= unvisited
        if not nxt.any():
            break
        dist[nxt] = step
        unvisited &= ~nxt
        frontier = nxt

    outlet = dist[-1]
    reached = outlet[outlet >= 0]
    if reached.size == 0:
        return float("nan")
    return float(reached.mean() / (depth - 1))


def evaluate_volume_physics(volume: np.ndarray, voxel_size: float = 1.0) -> dict[str, float]:
    """Compute the full scalar physics validation suite on a single 3D binary volume."""
    vol = validate_binary_volume(volume)
    sf = solid_fraction(vol)
    metrics: dict[str, float] = {
        "solid_fraction": sf,
        "porosity": 1.0 - sf,
    }
    metrics.update(specific_surface_area(vol, voxel_size=voxel_size))
    metrics.update(directional_chord_metrics(vol, phase=0))
    metrics.update(directional_chord_metrics(vol, phase=1))
    metrics.update(connectivity_and_percolation(vol))
    metrics["geodesic_tortuosity_z"] = geodesic_pore_tortuosity(vol, axis=0)
    metrics["geodesic_tortuosity_y"] = geodesic_pore_tortuosity(vol, axis=1)
    metrics["geodesic_tortuosity_x"] = geodesic_pore_tortuosity(vol, axis=2)
    return metrics


def summarize_metric_records(records: Sequence[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Aggregate `{mean, std, min, max, valid_n}` across multiple 3D realizations."""
    if not records:
        return {}
    keys = list(records[0].keys())
    summary: dict[str, dict[str, float]] = {}
    for k in keys:
        vals = np.array([r[k] for r in records], dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            summary[k] = {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan"), "valid_n": 0.0}
        else:
            summary[k] = {
                "mean": float(finite.mean()),
                "std": float(finite.std()),
                "min": float(finite.min()),
                "max": float(finite.max()),
                "valid_n": float(finite.size),
            }
    return summary


def export_openimpala_tiff(volume: np.ndarray, output_path: Path, pore_label: int = 0, solid_label: int = 1) -> Path:
    """Export a 3D multi-page `uint8` TIFF stack formatted for OpenImpala transport solvers."""
    vol = validate_binary_volume(volume)
    mapped = np.where(vol > 0, solid_label, pore_label).astype(np.uint8)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(output_path, mapped)
    return output_path
