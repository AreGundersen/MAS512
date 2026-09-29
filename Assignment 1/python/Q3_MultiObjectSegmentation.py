
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO

# Directories and paths
HERE      = Path(__file__).resolve().parent
videoDir  = HERE / "Q3"
inputVideo  = videoDir / "pedestrians.mp4"
outputVideo = videoDir / "q3_output.mp4"
modelPath = HERE / "yolo11n-seg.pt"     # Pretrained COCO segmentation model (downloaded automatically if missing)

# Parameters
TARGET_CLASS = 0        # COCO class id: 0 = person
confTresh    = 0.4      # Confidence threshold for detection (default in ultralytics is 0.25)
TRACKER      = "bytetrack.yaml"   # Tracking algorithm recommended by AI, alternative: "botsort.yaml" (slower, more robust to occlusion)
SHOW         = True     # Show live window while processing (press q to stop)


model = YOLO(modelPath)                                                     # Loading the segmentation model
cap = cv2.VideoCapture(str(inputVideo))                                     # Opening the input video (OpenCV wants a str, not Path)
if not cap.isOpened():
    raise FileNotFoundError(f"Could not open {inputVideo}")                 # Failing loudly instead of silently producing an empty output

w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))                                # Frame width in pixels
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))                               # Frame height in pixels
fps = cap.get(cv2.CAP_PROP_FPS) or 30                                       # Frames per second (fallback to 30 if the file does not report it)

REGION = np.array([[0, 0], [w, 0], [w, h // 2], [0, h // 2]], dtype=np.int32)   # Polygon for Q3 d): upper half of the frame, given as corner points (x, y)

writer = cv2.VideoWriter(str(outputVideo), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))  # Output video with the same size and fps as the input

allIds   = set()        # Every track ID seen in the whole video -> total number of unique persons
frameIdx = 0

while True:
    ok, frame = cap.read()                                                  # Reading the next frame
    if not ok:
        break                                                               # End of video
    frameIdx += 1

    # Q3 a) + c): segmentation and tracking in one call
    result = model.track(frame,
                         persist=True,                                      # Keep the tracker state between frames so IDs stay stable
                         tracker=TRACKER,
                         classes=[TARGET_CLASS],                            # Only detect the chosen class
                         conf=confTresh,
                         verbose=False)[0]                                  # track() returns a list with one Results object per frame

    annotated = result.plot()                                               # Drawing masks (a), boxes and track IDs (c) onto a copy of the frame

    nInFrame  = 0                                                           # Q3 b): objects in the scene in this frame
    nInRegion = 0                                                           # Q3 d): objects inside REGION in this frame

    if result.boxes is not None and result.boxes.id is not None:            # id is None if the tracker has not assigned IDs yet (e.g. no detections)
        boxes = result.boxes.xyxy.cpu().numpy()                             # Bounding boxes as (x1, y1, x2, y2) in pixels
        ids   = result.boxes.id.int().cpu().numpy()                         # Track ID for each box
        nInFrame = len(ids)                                                 # Number of detected persons in this frame
        allIds.update(ids.tolist())                                         # Remembering the IDs for the total count

        # Q3 d): count objects whose box centre lies inside the region
        for (x1, y1, x2, y2), tid in zip(boxes, ids):
            centerX, centerY = int((x1 + x2) / 2), int((y1 + y2) / 2)       # Using the box centre as the position of the person
            inside = cv2.pointPolygonTest(REGION, (centerX, centerY), False) >= 0   # +1 inside, 0 on the edge, -1 outside
            if inside:
                nInRegion += 1
            cv2.circle(annotated, (centerX, centerY), 5, (0, 255, 0) if inside else (0, 0, 255), -1)   # Green dot inside region, red outside
            cv2.putText(annotated, f"ID {tid}", (centerX + 8, centerY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # 3. Draw region and counters ------------------------------------------------------------------
    cv2.polylines(annotated, [REGION], isClosed=True, color=(255, 0, 255), thickness=3)   # Drawing the region polygon in magenta
    info = [f"Frame: {frameIdx}",
            f"(b) In scene now: {nInFrame}",
            f"(b) Unique tracked so far: {len(allIds)}",
            f"(d) In region: {nInRegion}"]
    for i, text in enumerate(info):
        cv2.putText(annotated, text, (20, 40 + 35 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)  # One text line per counter

    writer.write(annotated)                                                 # Writing the annotated frame to the output video
    if SHOW:
        cv2.imshow("Q3", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):                               # Press q to stop early
            break

# 4. Clean up --------------------------------------------------------------------------------------
cap.release()
writer.release()
cv2.destroyAllWindows()
print(f"Done. Frames: {frameIdx}, unique persons tracked: {len(allIds)}, saved to {outputVideo}")