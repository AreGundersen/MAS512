import cv2
import numpy as np

from pathlib import Path
from ultralytics import YOLO

# Universal path to the assignment folders
HERE = Path(__file__).resolve().parent
ASSIGNMENT1_DIR = HERE.parent / "Assignment 1"
MODEL_PATH = ASSIGNMENT1_DIR / "YOLO_swinging_load.pt"
RGB_PATH = ASSIGNMENT1_DIR / "Assignment 1-class" / "train" / "color_video" / "rgb_train_val_swingingload.avi"
DEPTH_DIR = ASSIGNMENT1_DIR / "Assignment 1-class" / "train" / "depth_raw"   # raw uint16 depth, one .npy per frame
OUTPUT_CSV = HERE / "Q1" / "BoxCenterXYZ.csv"

# Parameters
DEPTH_SCALE = 0.00025   # L515 raw depth units -> meters
MAX_RADIUS = 15         # Maximum radius in pixels to search for a valid depth around the box centre

# The zero-padded names (depth_0000_raw.npy ...) sort in frame order
depthFiles = sorted(DEPTH_DIR.glob("depth_*_raw.npy"))

# Check that the model, video and depth files exist
for p in (MODEL_PATH, RGB_PATH, DEPTH_DIR):
    print(p, ": OK" if p.exists() else ": MISSING")
print(f"Depth files found: {len(depthFiles)}")


def getDepth(depth, u, v):
    """Depth at (u, v). If 0 (no measurement), search outward until valid pixels are found."""
    h, w = depth.shape
    for r in range(MAX_RADIUS + 1):
        patch = depth[max(0, v - r):min(h, v + r + 1), max(0, u - r):min(w, u + r + 1)]
        valid = patch[patch > 0]
        if valid.size > 0:
            return np.median(valid) * DEPTH_SCALE, r     # r = 0 means the centre pixel itself
    return np.nan, -1                                    # nothing valid within MAX_RADIUS


OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)     # np.savetxt does not create folders

model = YOLO(str(MODEL_PATH))
capRgb = cv2.VideoCapture(str(RGB_PATH))

# The number of depth files must match the number of video frames
numberOfVideoFrames = int(capRgb.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video frames: {numberOfVideoFrames}")
assert len(depthFiles) == numberOfVideoFrames, "Depth file count differs from video frame count - check DEPTH_DIR"

rows = []   # (frame, cx, cy, Z, searchRadius)
for i, depthFile in enumerate(depthFiles):
    okRgb, rgb = capRgb.read()
    if not okRgb:
        break                                # video ended early

    depth = np.load(depthFile)               # (480, 640) uint16 in raw units

    if i == 0:
        # Sanity check of what we are working with
        print("RGB shape:", rgb.shape, "| depth shape:", depth.shape,
              "| depth dtype:", depth.dtype, "| depth max:", depth.max())

    result = model.predict(rgb, verbose=False)[0]
    if len(result.boxes) == 0:
        rows.append((i, np.nan, np.nan, np.nan, -1))     # no detection, keep the frame index
        continue

    best = result.boxes.conf.argmax()
    cx, cy, _, _ = result.boxes.xywh[best].tolist()

    scaleX = depth.shape[1] / rgb.shape[1]               # in case depth and RGB sizes differ
    scaleY = depth.shape[0] / rgb.shape[0]
    u = min(max(int(round(cx * scaleX)), 0), depth.shape[1] - 1)
    v = min(max(int(round(cy * scaleY)), 0), depth.shape[0] - 1)

    Z, radius = getDepth(depth, u, v)
    rows.append((i, cx, cy, Z, radius))

capRgb.release()

data = np.array(rows)   # columns: frame, cx, cy, Z, searchRadius
np.savetxt(OUTPUT_CSV, data, delimiter=",", header="frame,cx,cy,Z,searchRadius", comments="",
           fmt=["%d", "%.2f", "%.2f", "%.4f", "%d"])

print(f"Frames: {len(data)}, no detection: {np.isnan(data[:, 1]).sum()}, "
      f"no depth: {(np.isnan(data[:, 3]) & ~np.isnan(data[:, 1])).sum()}, "
      f"needed neighbours: {(data[:, 4] > 0).sum()}")