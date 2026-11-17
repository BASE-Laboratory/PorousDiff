"""RAM-cached datasets, multi-phase hierarchical encoding, symmetries, and RNG helpers."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

MATERIAL_TO_ID: dict[str, int] = {
    "virgin_carbon": 0,
    "tio2_anode": 1,
    "cuhcf_cathode": 2,
}
ID_TO_MATERIAL: dict[int, str] = {v: k for k, v in MATERIAL_TO_ID.items()}


def resolve_material_id(material: str | int) -> int:
    """Map canonical material name or integer index to `[0, 1, 2]`."""
    if isinstance(material, int):
        if material not in ID_TO_MATERIAL:
            raise ValueError(f"Unknown material_id {material}; valid={list(ID_TO_MATERIAL)}")
        return material
    key = material.strip().lower()
    if key in MATERIAL_TO_ID:
        return MATERIAL_TO_ID[key]
    raise ValueError(f"Unknown material {material!r}; choose from {list(MATERIAL_TO_ID)}")


def default_project_root() -> Path:
    """Resolve data/checkpoint root via `POROUSDIFF_ROOT` or sensible HPC fallback."""
    env = os.environ.get("POROUSDIFF_ROOT")
    if env:
        return Path(env).expanduser()
    hpc_default = Path("/projects/u6uc/asif")
    if hpc_default.exists():
        return hpc_default
    return Path.cwd()


def apply_compression_safe_symmetry(patch: np.ndarray, sym_id: int) -> np.ndarray:
    """Apply one of the 16 symmetries of `D_4(XY) x Z_2(Z)` preserving compression axis Z.

    Axis 0 (`Z`) is the vertical compression plunger axis; axes 1 and 2 (`Y, X`)
    span the isotropic carbon-felt sheet plane. Never swaps `Z` with `X` or `Y`.
    """
    code = int(sym_id) % 16
    rot_k = code & 0b11
    flip_y = bool((code >> 2) & 0b1)
    flip_z = bool((code >> 3) & 0b1)

    out = patch
    if rot_k:
        out = np.rot90(out, k=rot_k, axes=(1, 2))
    if flip_y:
        out = np.flip(out, axis=1)
    if flip_z:
        out = np.flip(out, axis=0)
    return np.ascontiguousarray(out)


def encode_phases_to_channels(patch: np.ndarray, num_phases: int = 2) -> torch.Tensor:
    """Encode `{0, 1}` or `{0, 1, 2}` `uint8` 3D volume into `{-1.0, +1.0}` float32 channels.

    For `num_phases == 3`, uses hierarchical 2-channel parameterization:
      - Channel 0 (`c_solid`): `+1` if `patch >= 1` (`carbon` OR `coating`), `-1` if `pore` (`0`).
      - Channel 1 (`c_coating`): `+1` if `patch == 2` (`active_coating`), `-1` otherwise.
    """
    if num_phases == 2:
        return (
            torch.from_numpy((patch > 0).astype(np.float32))
            .unsqueeze(0)
            .mul_(2.0)
            .sub_(1.0)
        )
    if num_phases == 3:
        solid = (patch >= 1).astype(np.float32) * 2.0 - 1.0
        coating = (patch == 2).astype(np.float32) * 2.0 - 1.0
        return torch.from_numpy(np.stack([solid, coating], axis=0))
    raise ValueError(f"Unsupported num_phases={num_phases}; expected 2 or 3")


def decode_channels_to_phases(
    tensor: torch.Tensor | np.ndarray,
    enforce_solid_support: bool = True,
    threshold: float = 0.0,
) -> np.ndarray:
    """Decode continuous diffusion output `[C, D, H, W]` or `[B, C, D, H, W]` to `uint8` phases."""
    arr = tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else np.asarray(tensor)
    if arr.ndim not in (4, 5):
        raise ValueError(f"Expected 4D [C,D,H,W] or 5D [B,C,D,H,W] array, got {arr.shape}")

    c_axis = 0 if arr.ndim == 4 else 1
    n_ch = arr.shape[c_axis]
    if n_ch == 1:
        slc = arr[0] if arr.ndim == 4 else arr[:, 0]
        return (slc > threshold).astype(np.uint8)
    if n_ch == 2:
        s_raw = arr[0] if arr.ndim == 4 else arr[:, 0]
        c_raw = arr[1] if arr.ndim == 4 else arr[:, 1]
        solid = s_raw > threshold
        coat = c_raw > threshold
        if enforce_solid_support:
            coat = coat & solid
        return (solid.astype(np.uint8) + coat.astype(np.uint8)).astype(np.uint8)
    raise ValueError(f"Unsupported channel count {n_ch}; expected 1 or 2")


def resolve_patch_split_dir(
    data_root: Path,
    state_name: str,
    patch_size: int,
    version: str,
    split: str,
) -> Path:
    """Resolve `<state>_{binary,multiphase}_patches<P>_<version>/<split>` directory."""
    candidates = [
        data_root / f"{state_name}_binary_patches{patch_size}_{version}" / split,
        data_root / f"{state_name}_multiphase_patches{patch_size}_{version}" / split,
    ]
    for cand in candidates:
        if cand.is_dir():
            return cand
    return candidates[0]


def load_split_patches(
    split_dir: Path,
    expected_shape: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Load `[N, D, H, W]` `uint8` patches from stacked `patches.npy` or individual `.npy` files."""
    stacked_path = split_dir / "patches.npy"
    if stacked_path.is_file():
        arr = np.load(stacked_path)
        if arr.ndim != 4 or arr.dtype != np.uint8:
            raise ValueError(f"Expected 4D uint8 array in {stacked_path}, got {arr.shape} {arr.dtype}")
        if expected_shape is not None and arr.shape[1:] != expected_shape:
            raise ValueError(f"Expected patch shape {expected_shape} in {stacked_path}, got {arr.shape[1:]}")
        return arr

    files = sorted(p for p in split_dir.glob("*.npy") if p.name != "patches.npy")
    if not files:
        raise RuntimeError(f"No patches found in {split_dir}")

    patches = []
    for path in files:
        p = np.load(path)
        if p.dtype != np.uint8:
            raise ValueError(f"Expected uint8 patch in {path}, got {p.dtype}")
        if expected_shape is not None and p.shape != expected_shape:
            raise ValueError(f"Expected patch shape {expected_shape} in {path}, got {p.shape}")
        patches.append(p)
    return np.stack(patches, axis=0)


class LoadConditionedPatchDataset(Dataset):
    """RAM-cached 3D microstructure dataset paired with continuous load values."""

    def __init__(
        self,
        data_root: Path,
        states: Mapping[str, float],
        split: str,
        patch_size: int = 96,
        version: str = "carbon_calibrated_v1",
        augment: bool = False,
        num_phases: int = 2,
    ) -> None:
        if split not in ("train", "val"):
            raise ValueError(f"Invalid split {split!r}; expected 'train' or 'val'")
        if not states:
            raise ValueError("states mapping must not be empty")

        self.data_root = Path(data_root)
        self.states = {str(k): float(v) for k, v in states.items()}
        self.split = split
        self.patch_size = patch_size
        self.version = version
        self.augment = augment
        self.num_phases = num_phases

        expected_shape = (patch_size, patch_size, patch_size)
        volume_blocks: list[np.ndarray] = []
        load_blocks: list[np.ndarray] = []
        self.state_counts: dict[str, int] = {}

        for state_name, load_val in self.states.items():
            split_dir = resolve_patch_split_dir(
                self.data_root, state_name, patch_size, version, split
            )
            arr = load_split_patches(split_dir, expected_shape=expected_shape)
            volume_blocks.append(arr)
            load_blocks.append(np.full(arr.shape[0], load_val, dtype=np.float32))
            self.state_counts[state_name] = int(arr.shape[0])

        self.patches = np.concatenate(volume_blocks, axis=0)
        self.loads = np.concatenate(load_blocks, axis=0)

    def __len__(self) -> int:
        return int(self.patches.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        patch = self.patches[index]
        if self.augment:
            sym_id = random.randrange(16)
            patch = apply_compression_safe_symmetry(patch, sym_id)

        image = encode_phases_to_channels(patch, num_phases=self.num_phases)
        load_val = torch.tensor(float(self.loads[index]), dtype=torch.float32)
        return image, load_val


class MultiElectrodePatchDataset(Dataset):
    """Unified RAM-cached dataset across Virgin, TiO2-Anode, and CuHCF-Cathode scans.

    Each entry in `scan_specs` specifies:
      - `state`: dataset state identifier
      - `load_value`: scalar load or calibrated compressive stress (`kPa` or `mV/V`)
      - `material`: `'virgin_carbon'`, `'tio2_anode'`, `'cuhcf_cathode'` (or int `0..2`)
      - `voxel_size_um`: effective voxel size in micrometers (default `1.0`)
      - `patch_dir` (optional): explicit patch directory override
    """

    def __init__(
        self,
        data_root: Path,
        scan_specs: Sequence[Mapping[str, object]],
        split: str,
        patch_size: int = 96,
        version: str = "carbon_calibrated_v1",
        num_phases: int = 3,
        augment: bool = False,
    ) -> None:
        if split not in ("train", "val"):
            raise ValueError(f"Invalid split {split!r}; expected 'train' or 'val'")
        if not scan_specs:
            raise ValueError("scan_specs must not be empty")

        self.data_root = Path(data_root)
        self.split = split
        self.patch_size = patch_size
        self.version = version
        self.num_phases = num_phases
        self.augment = augment

        expected_shape = (patch_size, patch_size, patch_size)
        vol_blocks: list[np.ndarray] = []
        load_blocks: list[np.ndarray] = []
        mat_blocks: list[np.ndarray] = []
        vox_blocks: list[np.ndarray] = []
        group_ids: list[np.ndarray] = []
        self.state_counts: dict[str, int] = {}
        self.material_counts: dict[str, int] = {}

        for group_idx, spec in enumerate(scan_specs):
            state_name = str(spec["state"])
            load_val = float(spec["load_value"])  # type: ignore[arg-type]
            mat_id = resolve_material_id(spec.get("material", "virgin_carbon"))  # type: ignore[arg-type]
            mat_name = ID_TO_MATERIAL[mat_id]
            vox_um = float(spec.get("voxel_size_um", 1.0))  # type: ignore[arg-type]

            if "patch_dir" in spec and spec["patch_dir"] is not None:
                split_dir = Path(str(spec["patch_dir"])) / split
            else:
                split_dir = resolve_patch_split_dir(
                    self.data_root, state_name, patch_size, version, split
                )

            arr = load_split_patches(split_dir, expected_shape=expected_shape)
            n = int(arr.shape[0])
            vol_blocks.append(arr)
            load_blocks.append(np.full(n, load_val, dtype=np.float32))
            mat_blocks.append(np.full(n, mat_id, dtype=np.int64))
            vox_blocks.append(np.full(n, vox_um, dtype=np.float32))
            group_ids.append(np.full(n, group_idx, dtype=np.int64))

            self.state_counts[state_name] = n
            self.material_counts[mat_name] = self.material_counts.get(mat_name, 0) + n

        self.patches = np.concatenate(vol_blocks, axis=0)
        self.loads = np.concatenate(load_blocks, axis=0)
        self.materials = np.concatenate(mat_blocks, axis=0)
        self.voxel_sizes = np.concatenate(vox_blocks, axis=0)
        self._group_ids = np.concatenate(group_ids, axis=0)

    def __len__(self) -> int:
        return int(self.patches.shape[0])

    def build_balanced_weights(self) -> torch.Tensor:
        """Compute per-sample weights giving equal gradient share to each material class
        and each state within a material (prevents 12-load Virgin from drowning 2-load Few-Shot)."""
        weights = np.zeros(len(self), dtype=np.float64)
        unique_mats = np.unique(self.materials)
        for mat_id in unique_mats:
            mat_mask = self.materials == mat_id
            mat_groups = np.unique(self._group_ids[mat_mask])
            for g_id in mat_groups:
                g_mask = mat_mask & (self._group_ids == g_id)
                count = int(np.count_nonzero(g_mask))
                if count > 0:
                    weights[g_mask] = 1.0 / (len(unique_mats) * len(mat_groups) * count)
        return torch.from_numpy(weights)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        patch = self.patches[index]
        if self.augment:
            sym_id = random.randrange(16)
            patch = apply_compression_safe_symmetry(patch, sym_id)

        image = encode_phases_to_channels(patch, num_phases=self.num_phases)
        load_val = torch.tensor(float(self.loads[index]), dtype=torch.float32)
        mat_id = torch.tensor(int(self.materials[index]), dtype=torch.long)
        vox_um = torch.tensor(float(self.voxel_sizes[index]), dtype=torch.float32)
        return image, load_val, mat_id, vox_um


def capture_rng_states() -> dict[str, object]:
    """Capture Python, NumPy, CPU Torch, and all CUDA Torch RNG states."""
    state: dict[str, object] = {
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["rng_torch_cuda"] = [s.cpu() for s in torch.cuda.get_rng_state_all()]
    return state


def restore_rng_states(state: Mapping[str, object]) -> None:
    """Restore RNG states saved by `capture_rng_states` (cross-device tolerant)."""
    if "rng_python" in state:
        random.setstate(state["rng_python"])  # type: ignore[arg-type]
    if "rng_numpy" in state:
        np.random.set_state(state["rng_numpy"])  # type: ignore[arg-type]
    if "rng_torch_cpu" in state:
        torch.set_rng_state(state["rng_torch_cpu"].cpu())  # type: ignore[union-attr]
    elif "rng_torch" in state:
        torch.set_rng_state(state["rng_torch"].cpu())  # type: ignore[union-attr]

    if torch.cuda.is_available() and "rng_torch_cuda" in state:
        saved_cuda = state["rng_torch_cuda"]
        if isinstance(saved_cuda, list) and len(saved_cuda) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all([s.cpu() for s in saved_cuda])
