import tifffile as tiff
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
import re

FOLDER = Path(r"C:\Users\nysad\OneDrive\Documents\GitHub\UVCameraThesis\UVCameraThesis\ArduCam App V1.0\static\captures\all_data")

exposures, means, medians, maxvals = [], [], [], []

for f in sorted(FOLDER.glob("*.dng")):
    print("Reading:", f)
    try:
        img = tiff.imread(f)

        # Try reading metadata
        exp = None
        with tiff.TiffFile(f) as tif:
            desc = tif.pages[0].tags["ImageDescription"].value
            if isinstance(desc, str):
                clean = desc.replace(". ", "").replace(".}", "}")
                try:
                    metadata = json.loads(clean)
                    exp = metadata.get("Exposure_ms", None)
                except Exception as e:
                    print("Metadata parse error:", e)

        # Fall back to filename
        if exp is None:
            match = re.search(r"_(\d+)ms_", f.name)
            if match:
                exp = int(match.group(1))

        mean_val = img.mean()
        median_val = np.median(img)
        max_val = img.max()

        exposures.append(exp)
        means.append(mean_val)
        medians.append(median_val)
        maxvals.append(max_val)

        print(f"{f.name}: Exposure={exp} ms, Mean={mean_val:.1f}, Median={median_val:.1f}, Max={max_val}")

    except Exception as e:
        print(f"Error reading {f}: {e}")

print("Collected exposures:", exposures)

# Plot results
if exposures:
    plt.figure(figsize=(8,5))
    plt.plot(exposures, means, "o-", label="Mean DN")
    plt.plot(exposures, medians, "s-", label="Median DN")
    plt.plot(exposures, maxvals, "x-", label="Max DN")
    plt.xlabel("Exposure time (ms)")
    plt.ylabel("Pixel value (DN)")
    plt.title("Exposure vs Pixel Brightness")
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("No exposures collected!")
