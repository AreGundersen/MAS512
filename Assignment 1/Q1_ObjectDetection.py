from pathlib import Path
import random
import shutil
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import io, color, filters
from ultralytics import YOLO

# Which steps to run. Annotate/dataset/train only need to run once.
DO_ANNOTATE = True
DO_DATASET  = True
DO_TRAIN    = True
DO_EVAL     = True      # Q1 a) b) c)

DEBUG = False            # show plots for annotation and evaluation

# Directories and paths
HERE = Path(__file__).resolve().parent
classDir   = HERE / "Assignment 1-class"
datasetDir = HERE / "Q1_work" / "dataset"
dataYaml   = HERE / "data.yaml"
modelPath  = HERE / "YOLO_swinging_load.pt"
outputDir  = HERE / "runs"
outputDir.mkdir(exist_ok=True)

# Annotation parameters for background subtraction
DIFF_THRESH = 55        # |I - B| above this is 'moving' (0-255)
SIGMA       = 1.5       # Gaussian smoothing of the difference image
RADIUS      = 70        # mask pixels farther than this from the centre are ignored

confTresh  = 0.5        # Confidence threshold for detection
depthScale = 0.00025    # L515: metres per raw depth unit (check against realsense_acq.ipynb)
random.seed(0)


def to_gray(path):
    img = io.imread(str(path))[..., :3]
    return img, color.rgb2gray(img).astype(np.float32) * 255  # M x N x 3 -> M x N, 0-255


def find_payload(gray, background):
    diff = filters.gaussian(np.abs(gray - background), sigma=SIGMA, preserve_range=True)
    rows, cols = np.where(diff > DIFF_THRESH)
    if rows.size == 0:
        return None, None
    center_x, center_y = cols.mean(), rows.mean()                       # Finding payload by finding the mean of the pixels of the picture
    near = np.hypot(cols - center_x, rows - center_y) < RADIUS          # Adding filter for outliers - removing pixels far from the load (rope, shadows)
    cols, rows = cols[near], rows[near]                                 # Applying the filter to the pixels
    center_x, center_y = int(cols.mean()), int(rows.mean())             # Recalculate the center after filtering
    x1, y1, x2, y2 = cols.min(), rows.min(), cols.max(), rows.max()     # box = extent of the mask
    return (center_x, center_y), (x1, y1, x2 - x1, y2 - y1)             # Returning the center and the box (x, y, w, h)


# 1. Annotate -> YOLO labels ------------------------------------------------------------------------
if DO_ANNOTATE:
    for split in ("train", "test"):
        framesDir, labelsDir = classDir / split / "rgb_frames", classDir / split / "labels"
        labelsDir.mkdir(exist_ok=True)
        files = sorted(framesDir.glob("*.png"))

        if any(labelsDir.glob("*.txt")):
            print(f"{split}: labels already exist ({len(list(labelsDir.glob('*.txt')))} files), skipping")  # Skipping labeling if labels already exist
            continue

        print(f"{split}: annotating...")
        background = np.median(np.stack([to_gray(f)[1] for f in files[::1]]), axis=0)   # Finding median background from every 5th frame (to avoid moving load)

        log = []
        for i, f in enumerate(files):
            img, gray = to_gray(f)                          # Getting the grayscale image
            H, W = gray.shape                               # Getting the height and width of the image
            center, box = find_payload(gray, background)    # Finding the center and the box of the load
            if box is None:
                log.append((np.nan, np.nan, np.nan, np.nan)); continue
            
            x, y, w, h = box                                                                                            # Getting the box coordinates
            (labelsDir / f"{f.stem}.txt").write_text(f"0 {(x + w/2)/W:.6f} {(y + h/2)/H:.6f} {w/W:.6f} {h/H:.6f}\n")    # Writing the YOLO label file with the class (0) and normalized coordinates of the box
            log.append((*center, w, h))

        log = np.array(log, dtype=float)                                                # Converting the log to a numpy array
        print(f"{split}: {(~np.isnan(log[:, 0])).sum()}/{len(files)} labels written")   # Showing how many labels were written out of the total number of frames

        if DEBUG:
            fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
            ax[0].plot(log[:, 0], label="center_x"); ax[0].plot(log[:, 1], label="center_y"); ax[0].legend(); ax[0].set_ylabel("px")
            ax[1].plot(log[:, 2], label="w");        ax[1].plot(log[:, 3], label="h");        ax[1].legend(); ax[1].set_ylabel("px")
            ax[1].set_xlabel("frame"); fig.suptitle(f"{split} annotation check"); plt.tight_layout(); plt.show()

# 2. Dataset folder + data.yaml ---------------------------------------------------------------------
if DO_DATASET:
    def copy(files, split):
        (datasetDir / "images" / split).mkdir(parents=True, exist_ok=True)  # Making the directories for the dataset images
        (datasetDir / "labels" / split).mkdir(parents=True, exist_ok=True)  # Making the directories for the dataset labels
        for img in files:
            shutil.copy(img, datasetDir / "images" / split / img.name)      # Copying the images to the dataset folder
            shutil.copy(img.parent.parent / "labels" / f"{img.stem}.txt", datasetDir / "labels" / split / f"{img.stem}.txt")  # Copying the labels to the dataset folder

    def labelled(split):    # Checking which frames have labels and returning the list of those frames
        return [f for f in sorted((classDir / split / "rgb_frames").glob("*.png"))
                if (classDir / split / "labels" / f"{f.stem}.txt").exists()] 

    train, test = labelled("train"), labelled("test")
    random.shuffle(train)
    nVal = int(0.2 * len(train))
    copy(train[nVal:], "train"); copy(train[:nVal], "val"); copy(test, "test")

    dataYaml.write_text(
        f"path: {datasetDir.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: Payload\n"
    )   # Writing the data.yaml file with the paths to the dataset and the class names

    print(f"dataset: train {len(train) - nVal}, val {nVal}, test {len(test)}")

# 3. Train -----------------------------------------------------------------------------------------
if DO_TRAIN:
    model = YOLO("yolo11n.pt")      # Loading the YOLO model
    model.train(data=dataYaml,  
                epochs=50, 
                imgsz=640, 
                batch=16, 
                patience=10, 
                project=outputDir, 
                name="swinging_load", 
                exist_ok=True)          # Parameters for training the YOLO model                  
    shutil.copy(outputDir / "swinging_load" / "weights" / "best.pt", modelPath) # Copying the best model weights to the modelPath
    print(f"Saved model to: {modelPath}")

# 4. Q1 a) b) c) -----------------------------------------------------------------------------------
if DO_EVAL:
    model = YOLO(modelPath)                         # Loading the trained YOLO model
    testFrames = classDir / "test" / "rgb_frames"   # Getting the test frames directory
    depthDir   = classDir / "test" / "depth_raw"    # Getting the depth frames directory
    files = sorted(testFrames.glob("*.png"))        # Getting the list of test frames

    def detect(frame):
        result = model.predict(frame, conf=confTresh, verbose=False)[0]
        if not len(result.boxes):
            return None
        box = result.boxes[result.boxes.conf.argmax()]
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        return int(x1), int(y1), int(x2), int(y2), int((x1 + x2) / 2), int((y1 + y2) / 2)

    # Q1 a) KPIs on the test set
    metrics = model.val(data=dataYaml, 
                        split="test", 
                        plots=True, 
                        project=outputDir, 
                        name="test_eval", 
                        exist_ok=True)
    print(f"Precision: {metrics.box.mp:.3f}   Recall: {metrics.box.mr:.3f}   mAP50: {metrics.box.map50:.3f}   mAP50-95: {metrics.box.map:.3f}")

    plots = sorted((outputDir / "test_eval").glob("*.png"))
    fig, ax = plt.subplots(2, (len(plots) + 1) // 2, figsize=(6 * ((len(plots) + 1) // 2), 10))
    for a, p in zip(ax.ravel(), plots):
        a.imshow(plt.imread(p)); a.set_title(p.stem); a.axis("off")
    for a in ax.ravel()[len(plots):]:
        a.axis("off")
    plt.tight_layout(); plt.show()
    
    # Q1 b) five random test images with bounding box and centre
    fig, ax = plt.subplots(1, 5, figsize=(22, 5))
    for a, f in zip(ax, random.sample(files, 5)):
        frame = cv2.imread(str(f))
        det = detect(frame)
        if det:
            x1, y1, x2, y2, center_x, center_y = det
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
            a.set_title(f"{f.name}\nbbox=({x1},{y1},{x2},{y2})  centre=({center_x},{center_y})", fontsize=8)
        else:
            a.set_title(f"{f.name}\nno detection", fontsize=8)
        a.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)); a.axis("off")
    plt.tight_layout(); plt.show()

    # Q1 c) centre and depth for all test frames
    depthFiles = sorted(depthDir.glob("*.npy"))
    assert len(depthFiles) == len(files), f"{len(depthFiles)} depth frames vs {len(files)} rgb frames"

    log = []                                                                # (frame, cx, cy, depth [m])
    for i, (f, d) in enumerate(zip(files, depthFiles)):
        det = detect(cv2.imread(str(f)))
        if det is None:
            log.append((i, np.nan, np.nan, np.nan)); continue
        _, _, _, _, center_x, center_y = det
        depth = np.squeeze(np.load(d))
        window = depth[max(0, center_y - 2):center_y + 3, max(0, center_x - 2):center_x + 3]   # 5x5 around the centre
        valid = window[window > 0]                                          # 0 = no measurement
        z = float(np.median(valid)) * depthScale if valid.size else np.nan
        log.append((i, center_x, center_y, z))

    log = np.array(log)
    np.savetxt(outputDir / "q1c_center_depth.csv", log, delimiter=",", fmt="%.3f", header="frame,center_x,center_y,depth_m", comments="")
    print(f"{len(files)} frames, load detected in {(~np.isnan(log[:, 1])).sum()}, depth valid in {(~np.isnan(log[:, 3])).sum()}")
    for row in log:
        print(f"frame {int(row[0]):4d}   centre=({row[1]:4.0f}, {row[2]:4.0f})   depth={row[3]:.3f} m")

    if DEBUG:
        fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        ax[0].plot(log[:, 0], log[:, 1], label="center_x"); ax[0].plot(log[:, 0], log[:, 2], label="center_y"); ax[0].set_ylabel("pixel"); ax[0].legend()
        ax[1].plot(log[:, 0], log[:, 3]); ax[1].set_ylabel("depth [m]"); ax[1].set_xlabel("frame")
        fig.suptitle("Bounding box centre and depth for all test frames"); plt.tight_layout(); plt.show()