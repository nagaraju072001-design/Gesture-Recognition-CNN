from camera import open_camera
import cv2

cap = open_camera()

print("\nTesting camera...\n")

frame_count = 0

while frame_count < 10:

    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame!")
        break

    frame_count += 1

    print(f"Frame {frame_count} captured successfully")
    print(f"Resolution: {frame.shape[1]} x {frame.shape[0]}")

cap.release()

print("\n✅ Camera test completed successfully!")