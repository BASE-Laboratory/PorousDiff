
from pathlib import Path
import numpy as np
import tifffile as tiff

# Project data directories
DATA_ROOT = Path("/projects/u6uc/asif/data")
INPUT_DIR = DATA_ROOT / "Op001_binary_corrected"
OUTPUT_DIR = DATA_ROOT / "Op001_patches64_binary"

PATCH_SIZE = 64
STRIDE_XY = 128
MAX_PATCHES = 200

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(INPUT_DIR.glob("*.tif"))
print(f"Found {len(files)} binary slices")

if len(files) < PATCH_SIZE:
    raise RuntimeError("Not enough slices for a 64³ patch")

# Load the first 64 slices for the initial experiment
volume = np.stack(
    [tiff.imread(f) for f in files[:PATCH_SIZE]],
    axis=0
)

# Verify binary convention
values = np.unique(volume)
print("Volume shape:", volume.shape)
print("Unique values:", values)

if not np.all(np.isin(values, [0, 1])):
    raise ValueError("Expected binary values 0 and 1")

count = 0

for y in range(0, volume.shape[1] - PATCH_SIZE + 1, STRIDE_XY):
    for x in range(0, volume.shape[2] - PATCH_SIZE + 1, STRIDE_XY):

        patch = volume[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]

        solid_fraction = patch.mean()

        # Exclude patches containing almost entirely one phase
        if not (0.01 < solid_fraction < 0.99):
            continue

        np.save(
            OUTPUT_DIR / f"op001_binary_patch_{count:05d}.npy",
            patch.astype(np.uint8)
        )

        count += 1

        if count >= MAX_PATCHES:
            break

    if count >= MAX_PATCHES:
        break

print(f"Saved {count} patches")
print("Output:", OUTPUT_DIR)
