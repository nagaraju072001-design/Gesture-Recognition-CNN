import cv2
import mediapipe as mp
from pathlib import Path

from config import DATASET_DIR, GESTURES, TOTAL_IMAGES

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

current_gesture = None
count = 0

print("\nPress:")
print("1 = Palm")
print("2 = Fist")
print("3 = Peace")
print("4 = Thumb Up")
print("5 = OK")
print("q = Quit\n")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    if result.multi_hand_landmarks:

        for hand in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

    key = cv2.waitKey(1) & 0xFF

    if key in GESTURES:

        current_gesture = GESTURES[key]

        gesture_path = DATASET_DIR / current_gesture
        gesture_path.mkdir(parents=True, exist_ok=True)

        count = len(list(gesture_path.glob("*.jpg")))

        print(f"\nCollecting {current_gesture}")

    if current_gesture is not None and count < TOTAL_IMAGES:

        filename = DATASET_DIR / current_gesture / f"{count:04}.jpg"

        cv2.imwrite(str(filename), frame)

        count += 1

        cv2.putText(
            frame,
            f"{current_gesture}: {count}/{TOTAL_IMAGES}",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

    cv2.imshow("Gesture Collector", frame)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()