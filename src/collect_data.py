import cv2
import mediapipe as mp
import os
import time

# ----------------------------
# DATASET SETTINGS
# ----------------------------

DATASET_PATH = "../dataset"

GESTURES = {
    ord('1'): "Palm",
    ord('2'): "Fist",
    ord('3'): "Peace",
    ord('4'): "Thumb_Up",
    ord('5'): "OK"
}

TOTAL_IMAGES = 500

# ----------------------------
# CREATE DATASET FOLDERS
# ----------------------------

for gesture in GESTURES.values():
    os.makedirs(os.path.join(DATASET_PATH, gesture), exist_ok=True)

# ----------------------------
# MEDIAPIPE HAND DETECTOR
# ----------------------------

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# ----------------------------
# OPEN CAMERA
# ----------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not found")
    exit()

print("\n==============================")
print(" Gesture Dataset Collection")
print("==============================")
print("Press")
print("1 -> Palm")
print("2 -> Fist")
print("3 -> Peace")
print("4 -> Thumb_Up")
print("5 -> OK")
print("Q -> Quit")
print("==============================\n")

current_gesture = None
image_count = 0

last_saved = time.time()

# --------------------------------------------------------
# MAIN LOOP
# --------------------------------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    # Draw instructions
    cv2.putText(frame,
                "1:Palm 2:Fist 3:Peace 4:Thumb_Up 5:OK",
                (10,30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2)

    if current_gesture is not None:

        cv2.putText(frame,
                    f"Gesture : {current_gesture}",
                    (10,65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255,0,0),
                    2)

        cv2.putText(frame,
                    f"Saved : {image_count}/{TOTAL_IMAGES}",
                    (10,100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,0,255),
                    2)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            h, w, c = frame.shape

            x_list = []
            y_list = []

            for lm in hand_landmarks.landmark:
                x_list.append(int(lm.x * w))
                y_list.append(int(lm.y * h))

            xmin = max(min(x_list) - 30, 0)
            ymin = max(min(y_list) - 30, 0)

            xmax = min(max(x_list) + 30, w)
            ymax = min(max(y_list) + 30, h)

            cv2.rectangle(frame,
                          (xmin, ymin),
                          (xmax, ymax),
                          (0,255,0),
                          2)

            hand_crop = frame[ymin:ymax, xmin:xmax]

            if hand_crop.size != 0:

                hand_crop = cv2.resize(hand_crop, (224,224))

                cv2.imshow("Hand Crop", hand_crop)

                if current_gesture is not None:

                    if time.time() - last_saved > 0.3:

                        filename = os.path.join(
                            DATASET_PATH,
                            current_gesture,
                            f"{current_gesture}_{image_count+1}.jpg"
                        )

                        cv2.imwrite(filename, hand_crop)

                        image_count += 1

                        last_saved = time.time()

                        if image_count >= TOTAL_IMAGES:

                            print(f"{current_gesture} completed!")

                            current_gesture = None

                            image_count = 0

 # -----------------------------
    # Keyboard Controls
    # -----------------------------

    key = cv2.waitKey(1) & 0xFF

    if key in GESTURES:

        current_gesture = GESTURES[key]
        image_count = len(os.listdir(os.path.join(DATASET_PATH, current_gesture)))

        print(f"\nCollecting images for {current_gesture}")
        print(f"Already saved: {image_count}")

    elif key == ord('q'):

        print("Exiting...")
        break

    # Show webcam
    cv2.imshow("Gesture Dataset Collection", frame)

# -----------------------------
# Release Resources
# -----------------------------

cap.release()
cv2.destroyAllWindows()
hands.close()

print("\nDataset collection finished successfully!")