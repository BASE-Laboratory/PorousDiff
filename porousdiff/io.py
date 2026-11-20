"""Layer 2 volume I/O bridges delegating calibrated OME-Zarr v0.4 and TIFF storage to `basecoat`."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import basecoat
from basecoat import UnitConversionError, Volume, VolumeSet
import numpy as np

DEFAULT_BINARY_LABELS: dict[int, str] = {
    0: "pore",
    1: "solid",
}

DEFAULT_MULTIPHASE_LABELS: dict[int, str] = {
    0: "pore",
    1: "carbon",
    2: "coating",
}


def to_basecoat_volume(
    volume: np.ndarray | Volume,
    *,
    voxel_size_um: float | Sequence[float] = 1.0,
    preserve_multiphase: bool = False,
    pore_label: int = 0,
    solid_label: int = 1,
    label_map: Mapping[int, str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Volume:
    """Wrap a 3D `(Z, Y, X)` microstructure array into a calibrated `basecoat.Volume` (`unit='um'`).

    Satisfies `OpenImpala v5.1+` and `SMS` `VolumeProtocol` (`spacing_in('m')`), eliminating
    uncalibrated raw-array unit mismatches at solver boundaries.
    """
    arr = np.asarray(volume.data) if isinstance(volume, Volume) else np.asarray(volume)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D volume (Z, Y, X), got shape {arr.shape}")

    if preserve_multiphase:
        mapped = np.clip(arr, 0, 255).astype(np.uint8)
        resolved_labels = dict(label_map) if label_map is not None else dict(DEFAULT_MULTIPHASE_LABELS)
    else:
        mapped = np.where(arr > 0, solid_label, pore_label).astype(np.uint8)
        resolved_labels = (
            dict(label_map)
            if label_map is not None
            else {int(pore_label): "pore", int(solid_label): "solid"}
        )

    if isinstance(voxel_size_um, (int, float, np.floating, np.integer)):
        vox = float(voxel_size_um)
        if vox <= 0.0 or not np.isfinite(vox):
            raise ValueError(f"voxel_size_um must be positive and finite, got {voxel_size_um!r}")
        spacing: float | Sequence[float] = vox
    else:
        vals = tuple(float(v) for v in voxel_size_um)
        if len(vals) != 3 or any(v <= 0.0 or not np.isfinite(v) for v in vals):
            raise ValueError(
                f"voxel_size_um sequence must have 3 positive finite values (dz, dy, dx), got {voxel_size_um!r}"
            )
        spacing = vals

    prov = {"producer": "porousdiff", **(dict(metadata) if metadata else {})}
    return Volume.from_array(
        mapped,
        voxel_size_um=spacing,
        label_names=resolved_labels,
        provenance=prov,
    )


def export_volume_ome_zarr(
    volume: np.ndarray | Volume,
    store_path: str | Path,
    *,
    voxel_size_um: float | Sequence[float] = 1.0,
    preserve_multiphase: bool = False,
    pore_label: int = 0,
    solid_label: int = 1,
    label_map: Mapping[int, str] | None = None,
    multiscale_levels: int = 2,
    chunks: tuple[int, int, int] = (64, 64, 64),
    metadata: Mapping[str, Any] | None = None,
    overwrite: bool = True,
) -> Path:
    """Export a calibrated OME-Zarr v0.4 store (`unit='um'`, mode-downsampled `labels/phase`).

    Replaces bespoke `export_openimpala_tiff` hand-offs: both `OpenImpala v5.1+` and `SMS`
    consume the resulting `.ome.zarr` store directly with calibrated SI spacing.
    """
    phase_vol = to_basecoat_volume(
        volume,
        voxel_size_um=voxel_size_um,
        preserve_multiphase=preserve_multiphase,
        pore_label=pore_label,
        solid_label=solid_label,
        label_map=label_map,
        metadata=metadata,
    )
    image_vol = Volume(
        data=phase_vol.data.astype(np.float32),
        axes=phase_vol.axes,
        provenance=phase_vol.provenance,
    )
    vset = VolumeSet(image=image_vol, labels={"phase": phase_vol})
    out = Path(store_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return basecoat.write_ome_zarr(
        vset,
        out,
        multiscale_levels=int(multiscale_levels),
        chunks=chunks,
        overwrite=overwrite,
    )


def load_volume_array(
    path: str | Path,
    *,
    label_key: str = "phase",
) -> tuple[np.ndarray, float | None]:
    """Load a 3D `(Z, Y, X)` `uint8` microstructure volume and optional calibrated `voxel_size_um`.

    Supports `.ome.zarr` / `.zarr` (selecting `labels[label_key]` when present), `.tif` / `.tiff`,
    `.h5` / `.hdf5` / `.nxs` via `basecoat`, and `.npy`.
    """
    p = Path(path)
    name_low = p.name.lower()

    if name_low.endswith(".npy"):
        arr = np.load(p, allow_pickle=False)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D volume (Z, Y, X), got shape {arr.shape}")
        return np.asarray(arr, dtype=np.uint8), None

    if name_low.endswith((".ome.zarr", ".zarr")):
        vset = basecoat.read_ome_zarr(p, level=0, as_set=True)
        vol = vset.labels[label_key] if label_key in vset.labels else vset.image
    else:
        loaded_any = basecoat.read_volume(p)
        vol = loaded_any.image if isinstance(loaded_any, VolumeSet) else loaded_any

    arr = np.asarray(vol.data)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D volume (Z, Y, X), got shape {arr.shape}")

    vox_um: float | None = None
    try:
        sp_um = vol.spacing_in("um")
        vox_um = float(np.mean(sp_um))
    except UnitConversionError:
        vox_um = None

    return np.asarray(arr, dtype=np.uint8), vox_um
