from camera import open_camera
import cv2

cap = open_camera()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("Camera Test", frame)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()