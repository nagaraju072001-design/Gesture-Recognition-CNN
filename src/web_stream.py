from flask import Flask, Response
import cv2
import mediapipe as mp
import numpy as np
import pickle

from tensorflow.keras.models import load_model

from camera import open_camera
from config import MODELS_DIR

# ---------------------------------------------------
# Load Model
# ---------------------------------------------------

print("Loading AI Model...")

model = load_model(MODELS_DIR / "gesture_model.keras")

with open(MODELS_DIR / "label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

print("Model Loaded Successfully!")

# ---------------------------------------------------
# Camera
# ---------------------------------------------------

cap = open_camera()

# ---------------------------------------------------
# MediaPipe
# ---------------------------------------------------

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

# ---------------------------------------------------
# Flask
# ---------------------------------------------------

app = Flask(__name__)


def generate():

    while True:

        success, frame = cap.read()

        if not success:
            continue

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                )

                data = []

                for lm in hand_landmarks.landmark:
                    data.extend([
                        lm.x,
                        lm.y,
                        lm.z
                    ])

                data = np.array(data, dtype=np.float32).reshape(1, -1)

                prediction = model.predict(data, verbose=0)

                class_id = np.argmax(prediction)

                confidence = float(prediction[0][class_id])

                gesture = label_encoder.inverse_transform([class_id])[0]

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

                cv2.putText(
                    frame,
                    f"{gesture} ({confidence*100:.1f}%)",
                    (x1, y1 - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

        ret, buffer = cv2.imencode(".jpg", frame)

        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + frame +
            b'\r\n'
        )


@app.route("/")
def index():

    return """
    <html>

    <head>

        <title>Gesture Recognition</title>

    </head>

    <body style="background:#111;color:white;text-align:center;">

        <h1>Gesture Recognition AI</h1>

        <img src="/video_feed" width="900">

    </body>

    </html>
    """


@app.route("/video_feed")
def video_feed():

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        threaded=True,
    )
