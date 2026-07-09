import cv2
import mediapipe as mp
import pandas as pd
from pathlib import Path

from config import DATASET_DIR

# ----------------------------------
# MediaPipe Hand Detector
# ----------------------------------

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# ----------------------------------
# Gesture folders
# ----------------------------------

GESTURES = [
    "Palm",
    "Fist",
    "Peace",
    "Thumb_Up",
    "OK"
]

# ----------------------------------
# CSV Columns
# ----------------------------------

columns = ["label"]

for i in range(21):
    columns.append(f"x{i}")
    columns.append(f"y{i}")
    columns.append(f"z{i}")

dataset = []

# ----------------------------------
# Process Images
# ----------------------------------

for gesture in GESTURES:

    gesture_path = DATASET_DIR / gesture

    images = list(gesture_path.glob("*.jpg"))

    print(f"\nProcessing {gesture} ({len(images)} images)")

    processed = 0

    for image_path in images:

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        if not results.multi_hand_landmarks:
            continue

        hand = results.multi_hand_landmarks[0]

        row = [gesture]

        for lm in hand.landmark:
            row.extend([
                lm.x,
                lm.y,
                lm.z
            ])

        dataset.append(row)

        processed += 1

    print(f"✓ Saved {processed} samples")

# ----------------------------------
# Save CSV
# ----------------------------------

df = pd.DataFrame(dataset, columns=columns)

output_file = DATASET_DIR / "landmarks.csv"

df.to_csv(output_file, index=False)

print("\n=================================")
print("Dataset Created Successfully")
print("=================================")
print(f"Total Samples : {len(df)}")
print(f"CSV Saved To  : {output_file}")