import cv2
import mediapipe as mp
from pathlib import Path

from config import DATASET_DIR

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5,
)

GESTURES = ["Palm", "Fist", "Peace", "Thumb_Up", "OK"]

for gesture in GESTURES:

    total = 0
    detected = 0

    print(f"\nChecking {gesture}")

    folder = DATASET_DIR / gesture

    for image_path in folder.glob("*.jpg"):

        total += 1

        image = cv2.imread(str(image_path))
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            detected += 1

    print(f"Total     : {total}")
    print(f"Detected  : {detected}")
    print(f"Rejected  : {total-detected}")