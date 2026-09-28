"""
Live YOLO inference on the screen.

Grabs the screen (or a region of it), runs the model, draws the boxes and shows the
number of detections in the top-left corner. Press q to quit.

Run:  python screen_inference.py                         (whole primary monitor)
      python screen_inference.py --model yolov8n.pt      (any other weights)
      python screen_inference.py --region 100 100 800 600  (x y w h of the screen area to watch)

Requires:  pip install mss ultralytics opencv-python
"""
import argparse
import time
import numpy as np
import cv2
import mss
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="YOLO_swinging_load.pt", help="path to weights")
ap.add_argument("--conf", type=float, default=0.4, help="confidence threshold")
ap.add_argument("--region", type=int, nargs=4, metavar=("X", "Y", "W", "H"),
                help="screen region to capture; default = whole primary monitor")
ap.add_argument("--scale", type=float, default=0.6, help="display window scale")
args = ap.parse_args()

model = YOLO(args.model)

with mss.mss() as sct:
    if args.region:
        x, y, w, h = args.region
        monitor = {"left": x, "top": y, "width": w, "height": h}
    else:
        monitor = sct.monitors[1]              # 1 = primary monitor (0 = all monitors combined)

    prev = time.time()
    while True:
        frame = np.array(sct.grab(monitor))[..., :3]          # BGRA -> BGR
        frame = np.ascontiguousarray(frame)

        results = model.predict(frame, conf=args.conf, verbose=False)[0]
        n = len(results.boxes)

        out = results.plot()                                  # boxes + labels drawn by ultralytics

        # counter in the top-left corner
        fps = 1 / max(time.time() - prev, 1e-6); prev = time.time()
        text = f"Detections: {n}"
        cv2.rectangle(out, (10, 10), (330, 90), (0, 0, 0), -1)
        cv2.putText(out, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(out, f"{fps:.1f} fps", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        if args.scale != 1.0:
            out = cv2.resize(out, None, fx=args.scale, fy=args.scale)
        cv2.imshow("YOLO screen inference (q = quit)", out)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()
