import cv2

from camera import open_camera
from hand_detector import HandDetector

cap = open_camera()

detector = HandDetector(max_num_hands=2)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame, hands = detector.detect(frame)

    cv2.putText(
        frame,
        f"Hands : {len(hands)}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    y = 80

    for hand in hands:

        cv2.putText(
            frame,
            f"{hand['type']} Hand",
            (20,y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,0,0),
            2
        )

        y += 35

    cv2.imshow("Professional Hand Detector", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()