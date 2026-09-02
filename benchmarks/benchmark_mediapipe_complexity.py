import time
import statistics

import cv2
import mediapipe as mp

WARMUP_FRAMES = 30
BENCHMARK_FRAMES = 100


def benchmark(model_complexity):

    print()
    print("=" * 60)
    print(f"MODEL COMPLEXITY: {model_complexity}")
    print("=" * 60)

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=model_complexity,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    # -----------------------------------------------------
    # Warm-up
    # -----------------------------------------------------

    print("Warm-up...")

    for _ in range(WARMUP_FRAMES):

        success, frame = cap.read()

        if not success:
            continue

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        hands.process(rgb)

    # -----------------------------------------------------
    # Benchmark
    # -----------------------------------------------------

    timings = []
    detected = 0

    print("Benchmarking...")

    for _ in range(BENCHMARK_FRAMES):

        success, frame = cap.read()

        if not success:
            continue

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        t0 = time.perf_counter()

        results = hands.process(rgb)

        t1 = time.perf_counter()

        timings.append(t1 - t0)

        if results.multi_hand_landmarks:
            detected += 1

    cap.release()
    hands.close()

    avg_ms = statistics.mean(timings) * 1000
    fps = 1000 / avg_ms

    print()
    print(f"Average MediaPipe : {avg_ms:.3f} ms")
    print(f"MediaPipe FPS     : {fps:.2f}")
    print(f"Frames with hands : {detected}/{BENCHMARK_FRAMES}")

    return avg_ms


print("=" * 60)
print("MEDIAPIPE MODEL COMPLEXITY COMPARISON")
print("=" * 60)

complexity_0 = benchmark(0)

complexity_1 = benchmark(1)

print()
print("=" * 60)
print("COMPARISON")
print("=" * 60)

print()
print(
    f"Complexity 0 : {complexity_0:.3f} ms "
    f"({1000 / complexity_0:.2f} FPS)"
)

print(
    f"Complexity 1 : {complexity_1:.3f} ms "
    f"({1000 / complexity_1:.2f} FPS)"
)

improvement = (
    (complexity_1 - complexity_0)
    / complexity_1
) * 100

print()
print(
    f"Potential speed improvement: "
    f"{improvement:.1f}%"
)

print()
print("=" * 60)
print("BENCHMARK COMPLETE")
print("=" * 60)
