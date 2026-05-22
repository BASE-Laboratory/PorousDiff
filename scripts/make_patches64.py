from pathlib import Path
import tifffile as tiff
import numpy as np

raw_folder = Path.home() / "PorousDiff/data/raw/Op001mVV"
out_folder = Path.home() / "PorousDiff/data/processed/patches64"
out_folder.mkdir(parents=True, exist_ok=True)

files = sorted(raw_folder.glob("*.tif"))

print("Total TIFF files:", len(files))

# use first 64 slices only for first test
volume = np.stack([tiff.imread(f) for f in files[:800]], axis=0)

print("Loaded volume:", volume.shape, volume.dtype)

# normalize 16-bit to 0-1
volume = volume.astype(np.float32) / 65535.0

PATCH = 64
STRIDE = 256   # not 64 for now, otherwise too many patches

patch_id = 0

zmax, ymax, xmax = volume.shape

for z in range(0, zmax - PATCH + 1, PATCH):
    for y in range(0, ymax - PATCH + 1, STRIDE):
        for x in range(0, xmax - PATCH + 1, STRIDE):
            patch = volume[z:z+PATCH, y:y+PATCH, x:x+PATCH]
            out_file = out_folder / f"op001_patch_{patch_id:05d}.npy"
            np.save(out_file, patch)
            patch_id += 1

print("Saved patches:", patch_id)
print("Output folder:", out_folder)
