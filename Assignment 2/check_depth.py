import cv2
import numpy as np

from pathlib import Path

# Path to the depth video (same layout as in Q1_Camera.py)
HERE = Path(__file__).resolve().parent
DEPTH_PATH = HERE.parent / "Assignment 1" / "Assignment 1-class" / "train" / "depth_video" / "depth_train_val_swingingload.avi"

# Which frame to inspect and where to save the preview image
FRAME_TO_INSPECT = 0
PREVIEW_PATH = HERE / "Q1" / "depth_preview.png"

capture = cv2.VideoCapture(str(DEPTH_PATH))
capture.set(cv2.CAP_PROP_POS_FRAMES, FRAME_TO_INSPECT)
success, frame = capture.read()
capture.release()
assert success, "Could not read the frame"

print("shape:", frame.shape, "| dtype:", frame.dtype)
print("min/max:", frame.min(), frame.max())

# If all three colour channels are identical, the video is plain grayscale.
# If they differ, the depth was stored with a colour map (e.g. rs.colorizer).
if frame.ndim == 3:
    channelsAreEqual = np.array_equal(frame[..., 0], frame[..., 1]) and np.array_equal(frame[..., 1], frame[..., 2])
    print("All channels equal (grayscale):", channelsAreEqual)

cv2.imwrite(str(PREVIEW_PATH), frame)