import sys
import time
import statistics

import cv2

sys.path.insert(0, "src")

from camera import open_camera
from hand_detector import HandDetector


WARMUP_FRAMES = 20
BENCHMARK_FRAMES = 100


def average_ms(values):
    return statistics.mean(values) * 1000


print("=" * 60)
print("MEDIAPIPE HAND DETECTION BENCHMARK")
print("=" * 60)

cap = open_camera()

detector = HandDetector(
    max_num_hands=2,
    detection_confidence=0.7,
    tracking_confidence=0.7,
)

print()
print("Warm-up...")
print()

for _ in range(WARMUP_FRAMES):

    success, frame = cap.read()

    if not success:
        continue

    frame = cv2.flip(frame, 1)

    detector.detect(
        frame,
        draw=False,
    )

print("Warm-up complete.")
print()

# ---------------------------------------------------------
# Benchmark WITHOUT drawing
# ---------------------------------------------------------

no_draw_times = []
hands_detected = 0

print("Benchmark 1: MediaPipe WITHOUT drawing")

for _ in range(BENCHMARK_FRAMES):

    success, frame = cap.read()

    if not success:
        continue

    frame = cv2.flip(frame, 1)

    t0 = time.perf_counter()

    frame, hands = detector.detect(
        frame,
        draw=False,
    )

    t1 = time.perf_counter()

    no_draw_times.append(t1 - t0)

    if hands:
        hands_detected += 1


# ---------------------------------------------------------
# Benchmark WITH drawing
# ---------------------------------------------------------

draw_times = []

print("Benchmark 2: MediaPipe WITH drawing")

for _ in range(BENCHMARK_FRAMES):

    success, frame = cap.read()

    if not success:
        continue

    frame = cv2.flip(frame, 1)

    t0 = time.perf_counter()

    frame, hands = detector.detect(
        frame,
        draw=True,
    )

    t1 = time.perf_counter()

    draw_times.append(t1 - t0)


cap.release()

# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

print()
print("=" * 60)
print("RESULTS")
print("=" * 60)

print()

if no_draw_times:

    no_draw = average_ms(no_draw_times)

    print(
        f"MediaPipe WITHOUT drawing : "
        f"{no_draw:.3f} ms"
    )

    print(
        f"Detection FPS             : "
        f"{1000 / no_draw:.2f}"
    )

if draw_times:

    draw = average_ms(draw_times)

    print(
        f"MediaPipe WITH drawing    : "
        f"{draw:.3f} ms"
    )

    print(
        f"Detection + drawing FPS   : "
        f"{1000 / draw:.2f}"
    )

if no_draw_times and draw_times:

    drawing_cost = (
        average_ms(draw_times)
        - average_ms(no_draw_times)
    )

    print()

    print(
        f"Estimated drawing cost    : "
        f"{drawing_cost:.3f} ms"
    )

print()

print(
    f"Frames with hands         : "
    f"{hands_detected}/{BENCHMARK_FRAMES}"
)

print()
print("=" * 60)
print("BENCHMARK COMPLETE")
print("=" * 60)
