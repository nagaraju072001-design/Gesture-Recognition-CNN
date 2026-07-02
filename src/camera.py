import cv2


def list_cameras(max_cameras=5):
    """
    Detect all available cameras.
    """
    cameras = []

    print("\nSearching for cameras...\n")

    for i in range(max_cameras):

        cap = cv2.VideoCapture(i)

        if cap.isOpened():

            ret, frame = cap.read()

            if ret:
                cameras.append(i)
                print(f"[{i}] Camera detected")

        cap.release()

    return cameras


def open_camera():

    cameras = list_cameras()

    if len(cameras) == 0:
        raise RuntimeError("No camera found!")

    print("\nAvailable Cameras")

    for cam in cameras:
        print(f"{cam} -> Camera {cam}")

    while True:

        choice = input("\nSelect Camera Number: ")

        try:

            choice = int(choice)

            if choice in cameras:

                cap = cv2.VideoCapture(choice)

                print(f"\nUsing Camera {choice}")

                return cap

        except:
            pass

        print("Invalid selection.")