"""Geometry-based numerical uniaxial Z-compression baselines from porous electrode literature.

Implements the two standard heuristics currently used in fuel-cell GDL and redox-flow
felt modelling when intermediate synchrotron XCT compression scans are unavailable:
  1. `affine_z_compress_to_target`: Uniaxial Z-coordinate rescaling with mass-conserving
     occupancy projection (affinely squashes Z without non-affine fiber bending/slip).
  2. `pillar_pore_deletion_compress`: Vertical 1-voxel pillar pore-deletion algorithm
     (e.g., ACS Appl. Energy Mater. 2025; GeoDict uniaxial geometrical compression).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import gaussian_filter


def _tile_slab_along_z(volume: np.ndarray, required_depth: int) -> np.ndarray:
    """Extend a 3D volume `(Z, Y, X)` to `required_depth` via mirror-bounce tiling along Z."""
    d = volume.shape[0]
    if d >= required_depth:
        return volume[:required_depth]
    if d == 1:
        return np.repeat(volume, required_depth, axis=0)

    period = np.concatenate([volume, volume[-2:0:-1]], axis=0)
    reps = math.ceil(required_depth / period.shape[0])
    return np.tile(period, (reps, 1, 1))[:required_depth]


def resolve_compression_ratio(
    initial_solid_fraction: float,
    target_solid_fraction: float | None = None,
    compression_ratio: float | None = None,
) -> float:
    """Resolve compressed-to-initial thickness ratio `H_comp / H_0` in `(0, 1]`."""
    if compression_ratio is not None:
        if not 0.05 <= compression_ratio <= 1.0:
            raise ValueError("compression_ratio (H_comp / H_0) must be in [0.05, 1.0]")
        return float(compression_ratio)
    if target_solid_fraction is None:
        raise ValueError("Provide either target_solid_fraction or compression_ratio")
    if not 0.0 < target_solid_fraction < 1.0:
        raise ValueError("target_solid_fraction must be in (0, 1)")
    if initial_solid_fraction <= 0.0:
        return 1.0
    # Mass conservation of incompressible solid fibers under uniaxial Z compression:
    # phi_target * H_comp = phi_0 * H_0  =>  H_comp / H_0 = phi_0 / phi_target
    return float(np.clip(initial_solid_fraction / target_solid_fraction, 0.08, 1.0))


def compute_mirror_tiled_fraction(
    source_depth: int,
    output_depth: int,
    initial_solid_fraction: float,
    target_solid_fraction: float | None = None,
    compression_ratio: float | None = None,
) -> float:
    """Return the fraction in `[0.0, 1.0)` of required uncompressed Z slices extended via mirror tiling.

    Equals `0.0` whenever `source_depth >= round(output_depth / ratio)` (i.e. when a tall
    real uncompressed sub-volume supplies the full required Z height without reflection).
    """
    ratio = resolve_compression_ratio(initial_solid_fraction, target_solid_fraction, compression_ratio)
    req_src_d = max(int(output_depth), int(round(output_depth / ratio)))
    if source_depth >= req_src_d or req_src_d <= 0:
        return 0.0
    return float(req_src_d - source_depth) / float(req_src_d)


def affine_z_compress_to_target(
    uncompressed_vol: np.ndarray,
    target_solid_fraction: float | None = None,
    compression_ratio: float | None = None,
    output_depth: int | None = None,
) -> np.ndarray:
    """Affinely rescale `Z` (axis 0) to simulate purely geometric uniaxial compression.

    Preserves both `{0, 1}` binary and `{0, 1, 2}` 3-phase labels while conserving
    solid/coating mass up to target solid fraction.
    """
    vol = np.asarray(uncompressed_vol, dtype=np.uint8)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume (Z, Y, X), got {vol.shape}")

    phi_0 = float((vol >= 1).mean())
    ratio = resolve_compression_ratio(phi_0, target_solid_fraction, compression_ratio)
    out_d = int(output_depth) if output_depth is not None else max(2, int(round(vol.shape[0] * ratio)))
    src_d = max(out_d, int(round(out_d / ratio)))
    slab = _tile_slab_along_z(vol, src_d)

    _, h, w = slab.shape
    edges = np.linspace(0.0, float(src_d), out_d + 1)
    occ_solid = np.zeros((out_d, h, w), dtype=np.float32)
    occ_coat = np.zeros((out_d, h, w), dtype=np.float32)
    has_coat = bool(np.any(slab == 2))

    for z_out in range(out_d):
        z0 = edges[z_out]
        z1 = edges[z_out + 1]
        i0 = int(math.floor(z0))
        i1 = min(src_d, int(math.ceil(z1)))
        span = max(z1 - z0, 1e-6)
        acc_s = np.zeros((h, w), dtype=np.float32)
        acc_c = np.zeros((h, w), dtype=np.float32)
        for iz in range(i0, i1):
            wt = float(min(z1, iz + 1.0) - max(z0, float(iz))) / span
            if wt > 0.0:
                acc_s += wt * (slab[iz] >= 1).astype(np.float32)
                if has_coat:
                    acc_c += wt * (slab[iz] == 2).astype(np.float32)
        occ_solid[z_out] = acc_s
        if has_coat:
            occ_coat[z_out] = acc_c

    expected_phi = (
        float(target_solid_fraction)
        if target_solid_fraction is not None
        else min(0.95, float((slab >= 1).mean()) / ratio)
    )
    n_vox = out_d * h * w
    target_s_count = int(round(expected_phi * n_vox))
    out = np.zeros((out_d, h, w), dtype=np.uint8)
    if target_s_count == 0:
        return out

    score_s = np.where(
        occ_solid > 0.0,
        occ_solid + 1e-2 * gaussian_filter(occ_solid, sigma=(0.6, 0.4, 0.4)),
        0.0,
    )
    k_s = min(target_s_count, int(np.count_nonzero(score_s > 0.0)))
    if k_s > 0:
        idx_s = np.argpartition(score_s.ravel(), -k_s)[-k_s:]
        out.ravel()[idx_s] = 1

    if has_coat:
        coat_ratio_0 = float((slab == 2).mean()) / max(float((slab >= 1).mean()), 1e-6)
        target_c_count = int(round(expected_phi * coat_ratio_0 * n_vox))
        score_c = np.where(
            (out == 1) & (occ_coat > 0.0),
            occ_coat + 1e-2 * gaussian_filter(occ_coat, sigma=(0.6, 0.4, 0.4)),
            0.0,
        )
        k_c = min(target_c_count, int(np.count_nonzero(score_c > 0.0)))
        if k_c > 0:
            idx_c = np.argpartition(score_c.ravel(), -k_c)[-k_c:]
            out.ravel()[idx_c] = 2

    return out


def pillar_pore_deletion_compress(
    uncompressed_vol: np.ndarray,
    target_solid_fraction: float | None = None,
    compression_ratio: float | None = None,
    output_depth: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Compress 3D electrode along Z by deleting pore voxels inside vertical 1-voxel pillars.

    Mirrors the standard geometrical compression heuristic in GDL/felt literature:
    each `(y, x)` column independently removes `(H_0 - H_comp)` pore voxels (`phase == 0`),
    preserving solid/coating voxel order along Z until column pores are exhausted.
    """
    vol = np.asarray(uncompressed_vol, dtype=np.uint8)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume (Z, Y, X), got {vol.shape}")

    gen = np.random.default_rng(42) if rng is None else rng
    phi_0 = float((vol >= 1).mean())
    ratio = resolve_compression_ratio(phi_0, target_solid_fraction, compression_ratio)
    out_d = int(output_depth) if output_depth is not None else max(2, int(round(vol.shape[0] * ratio)))
    src_d = max(out_d, int(round(out_d / ratio)))
    slab = _tile_slab_along_z(vol, src_d)
    n_delete = src_d - out_d
    if n_delete == 0:
        return slab.copy()

    _, h, w = slab.shape
    cols = slab.reshape(src_d, h * w)
    out_cols = np.empty((out_d, h * w), dtype=np.uint8)
    fallback_idx = np.linspace(0, src_d - 1, out_d).round().astype(np.int32)

    for c_idx in range(h * w):
        col = cols[:, c_idx]
        pore_pos = np.flatnonzero(col == 0)
        if pore_pos.size >= n_delete:
            drop = gen.choice(pore_pos, size=n_delete, replace=False)
            keep_mask = np.ones(src_d, dtype=bool)
            keep_mask[drop] = False
            out_cols[:, c_idx] = col[keep_mask]
        elif pore_pos.size > 0:
            solid_only = col[col > 0]
            sub_idx = np.linspace(0, solid_only.size - 1, out_d).round().astype(np.int32)
            out_cols[:, c_idx] = solid_only[sub_idx]
        else:
            out_cols[:, c_idx] = col[fallback_idx]

    return out_cols.reshape(out_d, h, w)
