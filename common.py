"""
common.py - shared settings and helpers used by collect.py, train.py and run.py.

All three scripts import from here, so training and live use always process
images in exactly the same way. If you change a setting here, re-collect or
re-train as needed.
"""
from pathlib import Path

import cv2
from skimage.feature import hog

# ---------------------------------------------------------------- settings
ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
MODEL_PATH = ROOT / "models" / "knn_model.pkl"
RESULTS_DIR = ROOT / "results"

CATEGORIES = ["bottle", "book", "cup"]   # folder names under dataset/train and dataset/test

CAMERA_INDEX = 0         # try 1 if the wrong camera opens
BOX_SIZE = 300           # side of the center box, in pixels
CROP_SIZE = 128          # crops are resized to CROP_SIZE x CROP_SIZE

K_VALUES = [1, 3, 5, 7]  # K values tried by train.py
SMOOTH_FRAMES = 10       # live labels are majority-voted over this many frames


# ------------------------------------------------------------------ camera
def open_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Close other apps using it, "
                           "or change CAMERA_INDEX in common.py.")
    return cap


def read_frame(cap):
    """Read one mirrored frame, or None if the camera fails."""
    ok, frame = cap.read()
    return cv2.flip(frame, 1) if ok else None


def box_coords(frame):
    """Corners (x1, y1, x2, y2) of the centered square box."""
    h, w = frame.shape[:2]
    size = min(BOX_SIZE, h, w)
    x1, y1 = (w - size) // 2, (h - size) // 2
    return x1, y1, x1 + size, y1 + size


def crop_box(frame):
    x1, y1, x2, y2 = box_coords(frame)
    return frame[y1:y2, x1:x2].copy()


# ---------------------------------------------------------------- features
def extract_features(crop_bgr):
    """Turn a cropped BGR image into a HOG feature vector (shape and edges only)."""
    crop = cv2.resize(crop_bgr, (CROP_SIZE, CROP_SIZE))
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)   # light denoise for webcam noise
    feats = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    return feats.astype("float32")