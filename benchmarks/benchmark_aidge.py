import sys
import time
import statistics

import cv2
import numpy as np

sys.path.insert(0, "src")

from camera import open_camera
from hand_detector import HandDetector
from normalize import LandmarkNormalizer
from aidge_model import AidgeGestureModel


WARMUP_FRAMES = 20
BENCHMARK_FRAMES = 300


def ms(values):
    return statistics.mean(values) * 1000


print("=" * 60)
print("AIDGE END-TO-END PIPELINE BENCHMARK")
print("=" * 60)

print()
print("Loading camera...")

cap = open_camera()

print("Loading MediaPipe...")

hand_detector = HandDetector(
    max_num_hands=2,
    detection_confidence=0.7,
    tracking_confidence=0.7,
)

print("Loading Aidge model...")

model = AidgeGestureModel()

normalizer = LandmarkNormalizer()

print()
print("Warm-up phase...")
print(f"Warm-up frames: {WARMUP_FRAMES}")
print(f"Benchmark frames: {BENCHMARK_FRAMES}")
print()

# ---------------------------------------------------------
# Warm-up
# ---------------------------------------------------------

for _ in range(WARMUP_FRAMES):

    success, frame = cap.read()

    if not success:
        continue

    frame = cv2.flip(frame, 1)

    frame, hands = hand_detector.detect(frame)

    for hand in hands:

        landmarks = []

        for lm in hand["landmarks"].landmark:

            landmarks.extend([
                lm.x,
                lm.y,
                lm.z,
            ])

        landmarks = normalizer.normalize(
            landmarks
        )

        model.predict(landmarks)


print("Warm-up complete.")
print()
print("Running benchmark...")
print()

# ---------------------------------------------------------
# Benchmark storage
# ---------------------------------------------------------

camera_times = []
mediapipe_times = []
normalization_times = []
aidge_times = []
total_times = []

frames_processed = 0
frames_with_hands = 0

start_total = time.perf_counter()

# ---------------------------------------------------------
# Benchmark
# ---------------------------------------------------------

for _ in range(BENCHMARK_FRAMES):

    frame_start = time.perf_counter()

    # -----------------------------------------------------
    # Camera capture
    # -----------------------------------------------------

    t0 = time.perf_counter()

    success, frame = cap.read()

    t1 = time.perf_counter()

    if not success:
        continue

    camera_times.append(t1 - t0)

    frame = cv2.flip(frame, 1)

    # -----------------------------------------------------
    # MediaPipe
    # -----------------------------------------------------

    t0 = time.perf_counter()

    frame, hands = hand_detector.detect(frame)

    t1 = time.perf_counter()

    mediapipe_times.append(t1 - t0)

    if hands:
        frames_with_hands += 1

    # -----------------------------------------------------
    # Process landmarks
    # -----------------------------------------------------

    for hand in hands:

        landmarks = []

        for lm in hand["landmarks"].landmark:

            landmarks.extend([
                lm.x,
                lm.y,
                lm.z,
            ])

        # -------------------------------------------------
        # Normalization
        # -------------------------------------------------

        t0 = time.perf_counter()

        landmarks = normalizer.normalize(
            landmarks
        )

        t1 = time.perf_counter()

        normalization_times.append(
            t1 - t0
        )

        # -------------------------------------------------
        # Aidge inference
        # -------------------------------------------------

        t0 = time.perf_counter()

        gesture, confidence = model.predict(
            landmarks
        )

        t1 = time.perf_counter()

        aidge_times.append(
            t1 - t0
        )

    # -----------------------------------------------------
    # Total frame time
    # -----------------------------------------------------

    frame_end = time.perf_counter()

    total_times.append(
        frame_end - frame_start
    )

    frames_processed += 1


end_total = time.perf_counter()

# ---------------------------------------------------------
# Cleanup
# ---------------------------------------------------------

cap.release()

# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

print("=" * 60)
print("RESULTS")
print("=" * 60)

print()

print(f"Frames processed        : {frames_processed}")
print(f"Frames with hands       : {frames_with_hands}")

print()

if camera_times:
    print(
        f"Camera capture          : "
        f"{ms(camera_times):.3f} ms"
    )

if mediapipe_times:
    print(
        f"MediaPipe detection    : "
        f"{ms(mediapipe_times):.3f} ms"
    )

if normalization_times:
    print(
        f"Normalization          : "
        f"{ms(normalization_times):.3f} ms"
    )

if aidge_times:
    print(
        f"Aidge inference        : "
        f"{ms(aidge_times):.3f} ms"
    )

if total_times:

    avg_total = ms(total_times)

    print(
        f"Total frame pipeline   : "
        f"{avg_total:.3f} ms"
    )

    print(
        f"Pipeline FPS           : "
        f"{1000 / avg_total:.2f}"
    )

print()

if aidge_times:

    avg_aidge = ms(aidge_times)

    print(
        f"Aidge model FPS        : "
        f"{1000 / avg_aidge:.2f}"
    )

elapsed = end_total - start_total

print(
    f"Wall-clock benchmark   : "
    f"{elapsed:.3f} s"
)

print()
print("=" * 60)
print("BENCHMARK COMPLETE")
print("=" * 60)
