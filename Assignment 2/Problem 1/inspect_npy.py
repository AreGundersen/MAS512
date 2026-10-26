import numpy as np

from pathlib import Path

# Folder that contains the Assignment 1 data (same layout as in Q1_Camera.py)
HERE = Path(__file__).resolve().parent
ASSIGNMENT1_DIR = HERE.parent / "Assignment 1"

# How many file names to print as a sample
NUMBER_OF_NAMES_TO_PRINT = 5

# Find every .npy file below the Assignment 1 folder, in a stable sorted order
npy_files = sorted(ASSIGNMENT1_DIR.rglob("*.npy"))
print(f"Found {len(npy_files)} .npy files")

if not npy_files:
    raise SystemExit("No .npy files found - check ASSIGNMENT1_DIR")

# Show where the files live and how they are named, so we can match them to video frames
print("Folder of first file:", npy_files[0].parent)
print("First names:", [file.name for file in npy_files[:NUMBER_OF_NAMES_TO_PRINT]])
print("Last name:", npy_files[-1].name)

# Inspect one file: is it raw sensor units (uint16) or already meters (float)?
first_depth = np.load(npy_files[0])
print("shape:", first_depth.shape, "| dtype:", first_depth.dtype)
print("min/max:", first_depth.min(), first_depth.max())

# Share of pixels without a measurement (depth = 0)
print("Fraction of zero pixels:", float(np.mean(first_depth == 0)))

# Typical value of the valid pixels, to judge the unit (meters vs. raw units)
valid_depths = first_depth[first_depth > 0]
if valid_depths.size > 0:
    print("Median of valid pixels:", float(np.median(valid_depths)))