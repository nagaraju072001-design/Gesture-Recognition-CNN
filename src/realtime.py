import cv2
import mediapipe as mp
import numpy as np
import pickle
import time

from collections import deque

from tensorflow.keras.models import load_model

from config import MODELS_DIR
from camera import open_camera

# -------------------------------------------------
# Load AI Model
# -------------------------------------------------

print("Loading AI Model...")

model = load_model(MODELS_DIR / "gesture_model.keras")

with open(MODELS_DIR / "label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

print("Model Loaded Successfully!")

# -------------------------------------------------
# Camera
# -------------------------------------------------

cap = open_camera()

# -------------------------------------------------
# MediaPipe
# -------------------------------------------------

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

print("\n======================================")
print(" Real-Time Gesture Recognition Started")
print(" Press Ctrl+C to stop")
print("======================================\n")

# -------------------------------------------------
# Variables
# -------------------------------------------------

last_gesture = ""
last_print_time = 0

history = deque(maxlen=5)

# -------------------------------------------------
# Main Loop
# -------------------------------------------------

try:

    while True:

        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        if not results.multi_hand_landmarks:
            history.clear()
            continue

        hand_landmarks = results.multi_hand_landmarks[0]

        # ------------------------------------------
        # Build Feature Vector
        # ------------------------------------------

        data = []

        for lm in hand_landmarks.landmark:

            data.extend([
                lm.x,
                lm.y,
                lm.z
            ])

        data = np.array(data, dtype=np.float32).reshape(1, 63)

        # ------------------------------------------
        # Prediction
        # ------------------------------------------

        prediction = model.predict(data, verbose=0)

        class_id = np.argmax(prediction)

        confidence = float(prediction[0][class_id])

        # Ignore weak predictions

        if confidence < 0.80:
            continue

        gesture = label_encoder.inverse_transform([class_id])[0]

        # ------------------------------------------
        # Prediction Smoothing
        # ------------------------------------------

        history.append(gesture)

        if len(history) < 5:
            continue

        stable_gesture = max(set(history), key=history.count)

        if history.count(stable_gesture) < 4:
            continue

        # ------------------------------------------
        # Print only when changed
        # ------------------------------------------

        now = time.time()

        if stable_gesture != last_gesture or (now - last_print_time) > 1:

            print(
                f"Gesture : {stable_gesture:<10} | Confidence : {confidence*100:6.2f}%"
            )

            last_gesture = stable_gesture
            last_print_time = now

except KeyboardInterrupt:

    print("\nStopping Gesture Recognition...")

finally:

    cap.release()

    hands.close()

    print("Camera Released.")
    print("Done.")
