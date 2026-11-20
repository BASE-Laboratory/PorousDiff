"""3D physical and morphological validation metrics for porous electrode volumes."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.ndimage import distance_transform_edt, label
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.stats import ks_2samp, wasserstein_distance

CAUCHY_CROFTON_3D_FACTOR: float = 2.0 / 3.0


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
    """Compute exposed pore-solid interfacial area per unit volume (`1 / voxel_size`).

    Returns both raw Manhattan voxel face counts (`ssa_total`) and the isotropic 3D
    Cauchy–Crofton `2/3`-corrected interfacial area (`ssa_total_cc`).
    """
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    vol = validate_binary_volume(volume)
    n_voxels = float(vol.size)

    faces_z = float(np.count_nonzero(vol[1:, :, :] != vol[:-1, :, :]))
    faces_y = float(np.count_nonzero(vol[:, 1:, :] != vol[:, :-1, :]))
    faces_x = float(np.count_nonzero(vol[:, :, 1:] != vol[:, :, :-1]))

    total_faces = faces_z + faces_y + faces_x
    scale = n_voxels * voxel_size
    ssa_raw = total_faces / scale
    return {
        "ssa_total": ssa_raw,
        "ssa_total_cc": ssa_raw * CAUCHY_CROFTON_3D_FACTOR,
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


def _slice_pair(length: int, delta: int) -> tuple[slice, slice]:
    if delta > 0:
        return slice(0, length - delta), slice(delta, length)
    if delta < 0:
        return slice(-delta, length), slice(0, length + delta)
    return slice(0, length), slice(0, length)


def geodesic_pore_tortuosity(
    volume: np.ndarray,
    axis: int = 0,
    connectivity: int = 26,
) -> float:
    """26-connected Euclidean shortest-path geometric tortuosity across `axis`.

    Uses exact `1.0`, `sqrt(2)`, and `sqrt(3)` voxel-center Euclidean step weights
    via compiled Cython `scipy.sparse.csgraph.dijkstra` (eliminating the `~40%` Manhattan
    staircase bias on diagonal `45 deg` channels while returning `nan` when non-percolating).
    """
    if connectivity not in (6, 26):
        raise ValueError(f"connectivity must be 6 or 26, got {connectivity}")
    vol = validate_binary_volume(volume)
    pore = np.moveaxis(vol == 0, axis, 0)
    depth, height, width = pore.shape
    if depth < 2:
        return 1.0
    if not (pore[0].any() and pore[-1].any()):
        return float("nan")

    idx_map = np.full(pore.shape, -1, dtype=np.int32)
    n_pore = int(np.count_nonzero(pore))
    idx_map[pore] = np.arange(n_pore, dtype=np.int32)

    inlet_ids = idx_map[0][idx_map[0] >= 0]
    outlet_ids = idx_map[-1][idx_map[-1] >= 0]
    if inlet_ids.size == 0 or outlet_ids.size == 0:
        return float("nan")

    forward_offsets: list[tuple[int, int, int, float]] = []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if (dz, dy, dx) == (0, 0, 0):
                    continue
                if dz < 0 or (dz == 0 and dy < 0) or (dz == 0 and dy == 0 and dx < 0):
                    continue
                norm_sq = dz * dz + dy * dy + dx * dx
                if connectivity == 6 and norm_sq > 1:
                    continue
                forward_offsets.append((dz, dy, dx, math.sqrt(float(norm_sq))))

    rows_list: list[np.ndarray] = [np.full(inlet_ids.shape, n_pore, dtype=np.int32)]
    cols_list: list[np.ndarray] = [inlet_ids]
    weights_list: list[np.ndarray] = [np.zeros(inlet_ids.shape, dtype=np.float32)]

    for dz, dy, dx, step_w in forward_offsets:
        sz0, sz1 = _slice_pair(depth, dz)
        sy0, sy1 = _slice_pair(height, dy)
        sx0, sx1 = _slice_pair(width, dx)
        u = idx_map[sz0, sy0, sx0]
        v = idx_map[sz1, sy1, sx1]
        valid = (u >= 0) & (v >= 0)
        if np.any(valid):
            u_v = u[valid]
            v_v = v[valid]
            rows_list.append(u_v)
            cols_list.append(v_v)
            weights_list.append(np.full(u_v.shape, step_w, dtype=np.float32))

    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    wts = np.concatenate(weights_list)
    graph = coo_matrix((wts, (rows, cols)), shape=(n_pore + 1, n_pore + 1)).tocsr()

    dists = dijkstra(csgraph=graph, directed=False, indices=n_pore)
    outlet_dists = dists[outlet_ids]
    reached = outlet_dists[np.isfinite(outlet_dists)]
    if reached.size == 0:
        return float("nan")
    return float(reached.mean() / float(depth - 1))


def euler_characteristic_3d(volume: np.ndarray, voxel_size: float = 1.0) -> dict[str, float]:
    """Exact 3D cubical Euler characteristic `chi = V - E + F - C` and volumetric density `chi_V`.

    Uses standard digital-topology duality (26-connected foreground / 6-connected background) on the
    active phase (`volume > 0`). Positive `chi` indicates isolated components or enclosed cavities;
    strongly negative `chi` indicates highly redundant fibrous loop/genus entanglement.
    """
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    vol = validate_binary_volume(volume)
    p = np.pad(vol > 0, 1, mode="constant", constant_values=False)

    c = int(np.count_nonzero(p[1:-1, 1:-1, 1:-1]))
    fz = int(np.count_nonzero(p[1:, 1:-1, 1:-1] | p[:-1, 1:-1, 1:-1]))
    fy = int(np.count_nonzero(p[1:-1, 1:, 1:-1] | p[1:-1, :-1, 1:-1]))
    fx = int(np.count_nonzero(p[1:-1, 1:-1, 1:] | p[1:-1, 1:-1, :-1]))
    f = fz + fy + fx

    ez = int(
        np.count_nonzero(
            p[1:-1, 1:, 1:] | p[1:-1, 1:, :-1] | p[1:-1, :-1, 1:] | p[1:-1, :-1, :-1]
        )
    )
    ey = int(
        np.count_nonzero(
            p[1:, 1:-1, 1:] | p[1:, 1:-1, :-1] | p[:-1, 1:-1, 1:] | p[:-1, 1:-1, :-1]
        )
    )
    ex = int(
        np.count_nonzero(
            p[1:, 1:, 1:-1] | p[1:, :-1, 1:-1] | p[:-1, 1:, 1:-1] | p[:-1, :-1, 1:-1]
        )
    )
    e = ez + ey + ex

    v = int(
        np.count_nonzero(
            p[1:, 1:, 1:]
            | p[1:, 1:, :-1]
            | p[1:, :-1, 1:]
            | p[1:, :-1, :-1]
            | p[:-1, 1:, 1:]
            | p[:-1, 1:, :-1]
            | p[:-1, :-1, 1:]
            | p[:-1, :-1, :-1]
        )
    )
    chi = float(v - e + f - c)
    phys_vol = float(vol.size) * (float(voxel_size) ** 3)
    return {
        "euler_characteristic": chi,
        "euler_density": chi / phys_vol,
    }


def pore_throat_radius_metrics(volume: np.ndarray, voxel_size: float = 1.0) -> dict[str, float]:
    """Compute inscribed 3D Euclidean Distance Transform (EDT) pore-radius quantiles (`r50`, `r90`)."""
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    vol = validate_binary_volume(volume)
    pore_mask = vol == 0
    if not np.any(pore_mask):
        return {
            "pore_radius_median_um": 0.0,
            "pore_radius_p90_um": 0.0,
        }
    if np.any(vol > 0):
        edt = distance_transform_edt(pore_mask) * float(voxel_size)
    else:
        edt = (
            distance_transform_edt(np.pad(pore_mask, 1, mode="constant", constant_values=False))[
                1:-1, 1:-1, 1:-1
            ]
            * float(voxel_size)
        )
    radii = edt[pore_mask]
    return {
        "pore_radius_median_um": float(np.percentile(radii, 50.0)),
        "pore_radius_p90_um": float(np.percentile(radii, 90.0)),
    }


def _count_tpb_edges_along_axis(vol: np.ndarray, axis: int) -> int:
    """Count voxel grid edges along `axis` whose 2x2 incident voxels contain {0, 1, 2}."""
    v = np.moveaxis(vol, axis, 0)
    # Incident 2x2 voxels around each interior grid line parallel to axis 0
    c00 = v[:, :-1, :-1]
    c01 = v[:, :-1, 1:]
    c10 = v[:, 1:, :-1]
    c11 = v[:, 1:, 1:]

    has0 = (c00 == 0) | (c01 == 0) | (c10 == 0) | (c11 == 0)
    has1 = (c00 == 1) | (c01 == 1) | (c10 == 1) | (c11 == 1)
    has2 = (c00 == 2) | (c01 == 2) | (c10 == 2) | (c11 == 2)
    return int(np.count_nonzero(has0 & has1 & has2))


def multiphase_interfacial_metrics(volume: np.ndarray, voxel_size: float = 1.0) -> dict[str, float]:
    """Compute 3-phase coated-electrode electrochemical descriptors (`0=pore, 1=carbon, 2=coating`).

    Returns:
      - `carbon_fraction`, `coating_fraction`, `coating_to_carbon_ratio`
      - `ssa_pore_coating`, `ssa_pore_coating_cc`: Electrochemically Active Surface Area (EASA, 1/voxel_size)
      - `ssa_pore_carbon`, `ssa_pore_carbon_cc`: Bare carbon-electrolyte surface area (1/voxel_size)
      - `ssa_carbon_coating`, `ssa_carbon_coating_cc`: Electronic collector/coating contact area (1/voxel_size)
      - `tpb_density`: Triple-Phase Boundary length per unit volume (1/voxel_size^2)
      - `wired_coating_fraction`: Fraction of active coating electrically wired to Z boundaries
      - `specific_z_pinch_excess`: Z-contact probability divided by `solid_fraction^2`
    """
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    vol = np.asarray(volume, dtype=np.uint8)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume (Z, Y, X), got shape {vol.shape}")

    n_voxels = float(vol.size)
    area_norm = n_voxels * voxel_size
    carbon_frac = float((vol == 1).mean())
    coat_frac = float((vol == 2).mean())
    solid_frac = carbon_frac + coat_frac

    def _pair_faces(a: int, b: int) -> float:
        total = 0
        for sl1, sl2 in (
            (vol[1:, :, :], vol[:-1, :, :]),
            (vol[:, 1:, :], vol[:, :-1, :]),
            (vol[:, :, 1:], vol[:, :, :-1]),
        ):
            total += int(np.count_nonzero(((sl1 == a) & (sl2 == b)) | ((sl1 == b) & (sl2 == a))))
        return float(total) / area_norm

    tpb_edges = (
        _count_tpb_edges_along_axis(vol, 0)
        + _count_tpb_edges_along_axis(vol, 1)
        + _count_tpb_edges_along_axis(vol, 2)
    )
    tpb_density = float(tpb_edges) / (n_voxels * (voxel_size**2))

    # Electrical percolation of active coating via 6-connected solid backbone (phase >= 1) to Z=0 or Z=D-1
    total_coat = int(np.count_nonzero(vol == 2))
    if total_coat == 0:
        wired_frac = 0.0
    else:
        struct6 = np.zeros((3, 3, 3), dtype=np.uint8)
        struct6[1, 1, :] = 1
        struct6[1, :, 1] = 1
        struct6[:, 1, 1] = 1
        solid_lbl, _ = label(vol >= 1, structure=struct6)
        boundary_ids = (set(np.unique(solid_lbl[0, :, :])) | set(np.unique(solid_lbl[-1, :, :]))) - {0}
        if not boundary_ids:
            wired_frac = 0.0
        else:
            wired_mask = np.isin(solid_lbl, list(boundary_ids)) & (vol == 2)
            wired_frac = float(np.count_nonzero(wired_mask) / total_coat)

    # 3D compressive pinch-point & solid contact coordination (links to K11 DIAD XRD microstrain)
    solid_mask = vol >= 1
    z_solid_faces = int(np.count_nonzero(solid_mask[1:, :, :] & solid_mask[:-1, :, :]))
    z_pinch_density = float(z_solid_faces) / area_norm
    n_z_pairs = max(1, (vol.shape[0] - 1) * vol.shape[1] * vol.shape[2])
    z_pair_prob = float(z_solid_faces) / float(n_z_pairs)
    specific_z_pinch_excess = (
        z_pair_prob / max(solid_frac**2, 1e-6) if solid_frac > 1e-6 else 0.0
    )

    if total_coat == 0 or vol.shape[0] < 3:
        coat_pinch_frac = 0.0
    else:
        sandwiched_coat = solid_mask[:-2, :, :] & (vol[1:-1, :, :] == 2) & solid_mask[2:, :, :]
        coat_pinch_frac = float(np.count_nonzero(sandwiched_coat) / total_coat)

    if total_coat == 0:
        mean_coat_coord = 0.0
    else:
        padded_solid = np.pad(solid_mask.astype(np.uint8), 1, mode="constant", constant_values=0)
        coord6 = (
            padded_solid[2:, 1:-1, 1:-1]
            + padded_solid[:-2, 1:-1, 1:-1]
            + padded_solid[1:-1, 2:, 1:-1]
            + padded_solid[1:-1, :-2, 1:-1]
            + padded_solid[1:-1, 1:-1, 2:]
            + padded_solid[1:-1, 1:-1, :-2]
        )
        mean_coat_coord = float(coord6[vol == 2].mean())

    ssa_pc = _pair_faces(0, 2)
    ssa_pcarb = _pair_faces(0, 1)
    ssa_cc = _pair_faces(1, 2)
    return {
        "carbon_fraction": carbon_frac,
        "coating_fraction": coat_frac,
        "coating_to_carbon_ratio": coat_frac / max(carbon_frac, 1e-8) if coat_frac > 0 else 0.0,
        "ssa_pore_coating": ssa_pc,
        "ssa_pore_coating_cc": ssa_pc * CAUCHY_CROFTON_3D_FACTOR,
        "ssa_pore_carbon": ssa_pcarb,
        "ssa_pore_carbon_cc": ssa_pcarb * CAUCHY_CROFTON_3D_FACTOR,
        "ssa_carbon_coating": ssa_cc,
        "ssa_carbon_coating_cc": ssa_cc * CAUCHY_CROFTON_3D_FACTOR,
        "tpb_density": tpb_density,
        "wired_coating_fraction": wired_frac,
        "z_pinch_contact_density": z_pinch_density,
        "specific_z_pinch_excess": specific_z_pinch_excess,
        "coating_pinch_fraction": coat_pinch_frac,
        "mean_coating_coordination": mean_coat_coord,
    }


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
    metrics.update(euler_characteristic_3d(vol, voxel_size=voxel_size))
    metrics.update(pore_throat_radius_metrics(vol, voxel_size=voxel_size))
    metrics["geodesic_tortuosity_z"] = geodesic_pore_tortuosity(vol, axis=0)
    metrics["geodesic_tortuosity_y"] = geodesic_pore_tortuosity(vol, axis=1)
    metrics["geodesic_tortuosity_x"] = geodesic_pore_tortuosity(vol, axis=2)
    metrics["geodesic_solid_tortuosity_z"] = geodesic_pore_tortuosity(1 - vol, axis=0)
    return metrics


def evaluate_multiphase_physics(volume: np.ndarray, voxel_size: float = 1.0) -> dict[str, float]:
    """Compute combined pore-transport + 3-phase electrochemical metrics on `{0, 1, 2}` volumes."""
    vol = np.asarray(volume, dtype=np.uint8)
    metrics = evaluate_volume_physics(vol >= 1, voxel_size=voxel_size)
    metrics.update(multiphase_interfacial_metrics(vol, voxel_size=voxel_size))
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


def compute_ensemble_distribution_distances(
    real_records: Sequence[Mapping[str, float]],
    pred_records: Sequence[Mapping[str, float]],
    keys: Sequence[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute Wasserstein-1 (`W_1`) and two-sample Kolmogorov-Smirnov (`KS`) ensemble distances."""
    if not real_records or not pred_records:
        return {}
    if keys is None:
        metric_keys = [k for k in real_records[0].keys() if k in pred_records[0]]
    else:
        metric_keys = list(keys)

    out: dict[str, dict[str, float]] = {}
    for k in metric_keys:
        r_vals = np.asarray([float(r.get(k, float("nan"))) for r in real_records], dtype=np.float64)
        p_vals = np.asarray([float(r.get(k, float("nan"))) for r in pred_records], dtype=np.float64)
        r_fin = r_vals[np.isfinite(r_vals)]
        p_fin = p_vals[np.isfinite(p_vals)]
        if r_fin.size == 0 or p_fin.size == 0:
            out[k] = {
                "wasserstein_1": float("nan"),
                "wasserstein_1_norm": float("nan"),
                "ks_stat": float("nan"),
                "ks_pvalue": float("nan"),
                "valid_n_real": float(r_fin.size),
                "valid_n_pred": float(p_fin.size),
            }
            continue
        w1 = float(wasserstein_distance(r_fin, p_fin))
        scale = max(float(np.std(r_fin)), abs(float(np.mean(r_fin))) * 0.05, 1e-6)
        ks_res = ks_2samp(r_fin, p_fin)
        out[k] = {
            "wasserstein_1": w1,
            "wasserstein_1_norm": float(w1 / scale),
            "ks_stat": float(ks_res.statistic),
            "ks_pvalue": float(ks_res.pvalue),
            "valid_n_real": float(r_fin.size),
            "valid_n_pred": float(p_fin.size),
        }
    return out


def export_openimpala_tiff(
    volume: np.ndarray,
    output_path: Path,
    pore_label: int = 0,
    solid_label: int = 1,
    preserve_multiphase: bool = False,
) -> Path:
    """Export a 3D multi-page `uint8` TIFF stack formatted for OpenImpala transport solvers.

    .. deprecated:: 0.5.2
       Prefer :func:`porousdiff.io.export_volume_ome_zarr`, which writes calibrated OME-Zarr v0.4
       stores (`unit='um'`, `labels/phase`) consumed directly by `OpenImpala v5.1+` and `SMS`.
    """
    import basecoat

    vol = np.asarray(volume)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume (Z, Y, X), got shape {vol.shape}")
    if preserve_multiphase:
        mapped = np.clip(vol, 0, 255).astype(np.uint8)
    else:
        mapped = np.where(vol > 0, solid_label, pore_label).astype(np.uint8)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    basecoat.write_volume( basecoat.Volume.from_array(mapped), output_path )
    return output_path
