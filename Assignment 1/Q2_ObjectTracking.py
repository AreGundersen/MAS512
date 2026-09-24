from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
videoPath = HERE / "Assignment 1-class" / "test" / "color_video" / "test_swingingload.avi"
modelPath = HERE / "YOLO_swinging_load.pt"
outputDir = HERE / "tracking"
outputDir.mkdir(exist_ok=True) 


model = YOLO(modelPath)             # Loading the trained YOLO model
confTresh = 0.5                     # Confidence threshold for detection
video = cv2.VideoCapture(videoPath) # Open the video file

if not video.isOpened():
    raise FileNotFoundError(f"Cannot open {videoPath}") # Recommended by AI to raise an error if the video cannot be opened, due to cv2 retruning None instead of raising an error

fps = video.get(cv2.CAP_PROP_FPS) or 30
W = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video: {videoPath}, {W}x{H} px, {fps:.1f} fps, model: {modelPath}, conf={confTresh}")
# -------------------------------------------------------------------

log = []            # (frame, t, cx, cy, w, h, conf)
trail = []          # centres drawn on the video
i = 0
while True:
    ok, frame = video.read()
    if not ok:
        break
        
    result = model.predict(frame, conf=confTresh, verbose=False)[0]

    if len(result.boxes):
        box = result.boxes[result.boxes.conf.argmax()]                                      # one load -> keep the most confident box
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()                                          # Fetching the lower left corner and upper right corner of the bounding box
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2                                   # Finding the centre of the bounding box
        log.append((i, i / fps, center_x, center_y, x2 - x1, y2 - y1, float(box.conf[0])))  # Recording the tracing data
    else:
        log.append((i, i / fps, np.nan, np.nan, np.nan, np.nan, np.nan))

    i += 1

video.release();  cv2.destroyAllWindows()

log = np.array(log)
np.savetxt(outputDir / "trajectory.csv", log, delimiter=",", fmt="%.3f",
           header="frame,t_s,center_x,center_y,w,h,conf", comments="")
det = ~np.isnan(log[:, 2])
print(f"{i} frames, load detected in {det.sum()} ({100 * det.mean():.1f} %)")
print(f"saved: {outputDir / 'trajectory.csv'}, {outputDir / 'trajectory.png'}")

fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(log[det, 2], log[det, 3], c=log[det, 0], cmap="viridis", s=12)
ax.plot(log[det, 2], log[det, 3], color="gray", lw=0.5, alpha=0.5)
pad = 30
ax.set_xlim(np.nanmin(log[:, 2]) - pad, np.nanmax(log[:, 2]) + pad)
ax.set_ylim(np.nanmax(log[:, 3]) + pad, np.nanmin(log[:, 3]) - pad)
ax.set_aspect("equal"); ax.set_xlabel("x [px]"); ax.set_ylabel("y [px]")
ax.set_title("Trajectory in the image plane")
fig.colorbar(sc, ax=ax, label="frame")
plt.tight_layout()
fig.savefig(outputDir / "trajectory.png", dpi=150)
plt.show()




