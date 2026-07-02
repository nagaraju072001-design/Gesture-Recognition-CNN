import os
from pathlib import Path

# -----------------------------
# Project Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
IMAGES_DIR = BASE_DIR / "images"

# Create folders automatically
DATASET_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)

# -----------------------------
# Gesture Classes
# -----------------------------
GESTURES = {
    ord("1"): "Palm",
    ord("2"): "Fist",
    ord("3"): "Peace",
    ord("4"): "Thumb_Up",
    ord("5"): "OK"
}

TOTAL_IMAGES = 500
IMAGE_SIZE = (224, 224)

# -----------------------------
# Model
# -----------------------------
MODEL_NAME = "gesture_model.keras"

# -----------------------------
# Camera
# -----------------------------
# -----------------------------
# Camera
# -----------------------------
CAMERA_INDEX = 1      # Default camera
SAVE_EVERY_N_FRAMES = 5
WINDOW_NAME = "Gesture Collector"

print("Configuration Loaded Successfully")