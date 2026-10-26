from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# Directories and paths
HERE           = Path(__file__).resolve().parent.parent
assignment1Dir = HERE.parent / "Assignment 1"
modelPath      = assignment1Dir / "YOLO_swinging_load.pt"                                            # YOLO model trained in Assignment 1
rgbPath        = assignment1Dir / "Assignment 1-class" / "train" / "color_video" / "rgb_train_val_swingingload.avi"
depthDir       = assignment1Dir / "Assignment 1-class" / "train" / "depth_raw"                       # One depth_*_raw.npy file per frame
outputCsv      = HERE  / "Problem 1"  / "Q1_output" / "BoxCenterXYZ.csv"                                     # Result: box centre pixel and depth per frame

# Parameters
depthScale = 0.00025    # L515 raw depth units -> meters
maxRadius  = 3          # Maximum radius in pixels to search for a valid depth around the box centre

# Check that the model, video and depth files exist
depthFiles = sorted(depthDir.glob("depth_*_raw.npy"))                       # Sorted, so file number i belongs to frame i
for p in (modelPath, rgbPath, depthDir):
    print(p, ": OK" if p.exists() else ": MISSING")
print(f"Depth files found: {len(depthFiles)}")

model    = YOLO(str(modelPath))                                             # Loading the trained YOLO model
capRgb   = cv2.VideoCapture(str(rgbPath))                                   # Opening the colour video (OpenCV wants a str, not Path)

rows = []                                                                   # One row per frame: (frame, cx, cy, Z, searchRadius)
i = 0
while i < len(depthFiles):
    okRgb, rgb = capRgb.read()                                              # Reading the next colour frame
    if not okRgb:
        break                                                               # End of video
    depth = np.load(depthFiles[i])                                          # Raw depth image of the same frame, (480, 640) uint16

    if i == 0:
        print("RGB shape:", rgb.shape, "| depth shape:", depth.shape, "| depth dtype:", depth.dtype, "| depth min/max:", depth.min(), depth.max())   # Sanity check

    result = model.predict(rgb, verbose=False)[0]                           # Detecting the load
    if len(result.boxes) == 0:
        rows.append((i, np.nan, np.nan, np.nan, -1))                        # No detection, keeping the frame index
        i += 1
        continue

    best = result.boxes.conf.argmax()                                       # Using the most confident box
    cx, cy, _, _ = result.boxes.xywh[best].tolist()                         # Box centre in pixels

    scaleX = depth.shape[1] / rgb.shape[1]                                  # In case depth and colour image have different sizes
    scaleY = depth.shape[0] / rgb.shape[0]
    u = min(max(int(round(cx * scaleX)), 0), depth.shape[1] - 1)            # Box centre in depth pixels, clipped to the image
    v = min(max(int(round(cy * scaleY)), 0), depth.shape[0] - 1)

    # Depth at the box centre. If the centre pixel has no measurement (0), search in growing squares around it
    Z, radius = np.nan, -1                                                  # Stays NaN / -1 if nothing valid is found within maxRadius
    for r in range(maxRadius + 1):
        patch = depth[max(0, v - r):min(depth.shape[0], v + r + 1), max(0, u - r):min(depth.shape[1], u + r + 1)]   # (2r+1) x (2r+1) square around the centre
        valid = patch[patch > 0]                                            # Only pixels with a measurement
        if valid.size > 0:
            Z, radius = np.median(valid) * depthScale, r                    # Median is robust against single bad pixels; r = 0 means the centre pixel itself
            break

    rows.append((i, cx, cy, Z, radius))
    i += 1

capRgb.release()

outputCsv.parent.mkdir(parents=True, exist_ok=True)                         # Creating the Q1 folder if it does not exist
data = np.array(rows)                                                       # Columns: frame, cx, cy, Z, searchRadius
np.savetxt(outputCsv, data, delimiter=",", header="frame,cx,cy,Z,searchRadius", comments="",
           fmt=["%d", "%.2f", "%.2f", "%.4f", "%d"])

print(f"Frames: {len(data)}, no detection: {np.isnan(data[:, 1]).sum()}, "
      f"no depth: {(np.isnan(data[:, 3]) & ~np.isnan(data[:, 1])).sum()}, "
      f"needed neighbours: {(data[:, 4] > 0).sum()}")
