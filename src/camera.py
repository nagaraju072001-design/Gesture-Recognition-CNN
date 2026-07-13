import cv2


def open_camera():
    """
    Opens the Logitech webcam connected to the Raspberry Pi.
    """

    print("\nOpening Logitech Camera...")

    # Open the webcam using the V4L2 backend
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

    if not cap.isOpened():
        raise RuntimeError("Cannot open Logitech camera!")

    # Optional: Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Test that we can actually read a frame
    ret, frame = cap.read()

    if not ret:
        cap.release()
        raise RuntimeError("Camera opened but could not read a frame!")

    print("✅ Logitech Camera opened successfully!")
    print(f"Resolution: {frame.shape[1]} x {frame.shape[0]}")

    return cap