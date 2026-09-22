from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import cv2
from ultralytics import solutions
import numpy as np

REPO = Path("/home/coder/object-detection")          

 
ASSIGNMENT = REPO / "Assignments" / "Assignment 1"
 
TRAIN_SESSION = ASSIGNMENT / "Assignment 1-class" / "train"
TEST_SESSION = ASSIGNMENT / "Assignment 1-class" / "test"
 
ROOT = REPO / "Assignments"
DATASET = ROOT / "dataset"               
MODEL_OUT = ROOT / "YOLO_swinging_load.pt"
RUNS = ROOT / "runs"


