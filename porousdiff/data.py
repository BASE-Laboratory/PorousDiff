"""RAM-cached datasets, anisotropy-preserving 3D symmetries, and full RNG state helpers."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch.utils.data import Dataset


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
    """RAM-cached 3D binary microstructure dataset paired with continuous load values."""

    def __init__(
        self,
        data_root: Path,
        states: Mapping[str, float],
        split: str,
        patch_size: int = 96,
        version: str = "carbon_calibrated_v1",
        augment: bool = False,
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

        expected_shape = (patch_size, patch_size, patch_size)
        volume_blocks: list[np.ndarray] = []
        load_blocks: list[np.ndarray] = []
        self.state_counts: dict[str, int] = {}

        for state_name, load_val in self.states.items():
            split_dir = (
                self.data_root
                / f"{state_name}_binary_patches{patch_size}_{version}"
                / split
            )
            arr = load_split_patches(split_dir, expected_shape=expected_shape)
            volume_blocks.append(arr)
            load_blocks.append(np.full(arr.shape[0], load_val, dtype=np.float32))
            self.state_counts[state_name] = int(arr.shape[0])

        # Compact contiguous uint8 storage in CPU RAM (~0.88 MB per 96^3 patch)
        self.patches = np.concatenate(volume_blocks, axis=0)
        self.loads = np.concatenate(load_blocks, axis=0)

    def __len__(self) -> int:
        return int(self.patches.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        patch = self.patches[index]
        if self.augment:
            sym_id = random.randrange(16)
            patch = apply_compression_safe_symmetry(patch, sym_id)

        # Map {0, 1} -> {-1.0, +1.0} float32
        image = torch.from_numpy(patch).to(dtype=torch.float32).unsqueeze(0).mul_(2.0).sub_(1.0)
        load_val = torch.tensor(float(self.loads[index]), dtype=torch.float32)
        return image, load_val


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
