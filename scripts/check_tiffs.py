from pathlib import Path
import tifffile as tiff

folder = Path.home() / "PorousDiff/data/raw/Op001mVV"
files = sorted(folder.glob("*.tif"))

print("Total files:", len(files))

bad = []

for f in files:
    try:
        img = tiff.imread(f)
        print("OK:", f.name, img.shape)
    except Exception as e:
        print("BAD:", f.name, e)
        bad.append(f.name)

print("Total bad files:", len(bad))

with open(folder / "bad_files.txt", "w") as out:
    for name in bad:
        out.write(name + "\n")
