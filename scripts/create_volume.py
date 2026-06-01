from pathlib import Path
import tifffile as tiff
import numpy as np

folder = Path.home() / "PorousDiff/data/raw/Op001mVV"
files = sorted(folder.glob("*.tif"))

print("Total files:", len(files))

# Load first 64 slices
slices = [tiff.imread(f) for f in files[:64]]

volume = np.stack(slices, axis=0)

print("Volume shape:", volume.shape)
print("Dtype:", volume.dtype)
print("Min:", volume.min())
print("Max:", volume.max())