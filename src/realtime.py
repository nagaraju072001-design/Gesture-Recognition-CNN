import cv2
import mediapipe as mp
import numpy as np
import pickle

from tensorflow.keras.models import load_model

from config import MODELS_DIR
from camera import open_camera

# ---------------------------------------
# Load Model and Label Encoder
# ---------------------------------------
print("Loading AI Model...")

model = load_model(MODELS_DIR / "gesture_model.keras")

with open(MODELS_DIR / "label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

print("Model Loaded Successfully!")

# ---------------------------------------
# Open Camera
# ---------------------------------------
cap = open_camera()

# ---------------------------------------
# MediaPipe Hands
# ---------------------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

# ---------------------------------------
# FPS
# ---------------------------------------
prev_time = cv2.getTickCount()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    total_hands = 0

    if results.multi_hand_landmarks:

        total_hands = len(results.multi_hand_landmarks)

        for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness):

            # Draw landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # ---------------------------------------
            # Normalize Landmarks (same as training)
            # ---------------------------------------
            wrist = hand_landmarks.landmark[0]

            data = []

            for lm in hand_landmarks.landmark:

                data.extend([
                    lm.x - wrist.x,
                    lm.y - wrist.y,
                    lm.z - wrist.z
                ])

            data = np.array(data, dtype=np.float32).reshape(1, -1)

            # ---------------------------------------
            # Predict Gesture
            # ---------------------------------------
            prediction = model.predict(data, verbose=0)

            class_id = np.argmax(prediction)

            confidence = float(prediction[0][class_id])

            gesture = label_encoder.inverse_transform([class_id])[0]

            # -----------------------------
            # Display probabilities on screen
            # -----------------------------
            start_y = 120

            for i, (name, prob) in enumerate(zip(label_encoder.classes_, prediction[0])):

                cv2.putText(
                    frame,
                    f"{name}: {prob*100:.1f}%",
                    (20, start_y + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )

            # ---------------------------------------
            # Hand Label
            # ---------------------------------------
            hand_label = handedness.classification[0].label

            # ---------------------------------------
            # Bounding Box
            # ---------------------------------------
            h, w, _ = frame.shape

            xs = [lm.x for lm in hand_landmarks.landmark]
            ys = [lm.y for lm in hand_landmarks.landmark]

            x1 = int(min(xs) * w)
            y1 = int(min(ys) * h)
            x2 = int(max(xs) * w)
            y2 = int(max(ys) * h)

            cv2.rectangle(
                frame,
                (x1 - 20, y1 - 20),
                (x2 + 20, y2 + 20),
                (0, 255, 0),
                2,
            )

            # ---------------------------------------
            # Display Text
            # ---------------------------------------
            cv2.putText(
                frame,
                hand_label,
                (x1, y1 - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2,
            )

            cv2.putText(
                frame,
                gesture,
                (x1, y1 - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"{confidence*100:.1f}%",
                (x1, y2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

    # ---------------------------------------
    # FPS
    # ---------------------------------------
    current = cv2.getTickCount()

    fps = cv2.getTickFrequency() / (current - prev_time)

    prev_time = current

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2,
    )

    cv2.putText(
        frame,
        f"Hands: {total_hands}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    cv2.imshow("Real-Time Gesture Recognition", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()