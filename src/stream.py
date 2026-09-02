import time
import threading

import cv2

from camera import open_camera
from detector import GestureDetector
from overlay import Overlay
from performance import PerformanceMonitor
from history import GestureHistory
from recorder import VideoRecorder


class VideoStream:

    def __init__(self):

        print("Starting Video Stream...")

        # -------------------------------------------------
        # Camera
        # -------------------------------------------------
        self.cap = open_camera()

        # -------------------------------------------------
        # Gesture Detector
        #
        # GestureDetector owns the ONLY UART connection.
        # -------------------------------------------------
        self.detector = GestureDetector()

        # -------------------------------------------------
        # Dashboard Components
        # -------------------------------------------------
        self.overlay = Overlay()

        self.performance = PerformanceMonitor()

        self.history = GestureHistory()

        self.recorder = VideoRecorder()

        self.last_camera_check = time.time()

        # -------------------------------------------------
        # Latest Dashboard Information
        # -------------------------------------------------
        self.status_lock = threading.Lock()

        self.status = {

            "fps": 0.0,

            "cpu": 0.0,

            "ram": 0.0,

            "hands": 0,

            "gesture": "None",

            "confidence": 0.0,

            "hand": "None",

            "command": "--",

            "packet": "--",

            "uart_status": "READY",

            "model_status": "LOADED",

            "camera_status": "CONNECTED",
            "aidge_ms": 0.0,
            "hand_gestures": {
                "Left": {"gesture": "None", "confidence": 0.0},
                "Right": {"gesture": "None", "confidence": 0.0},
            },

            "history": [],
        }

        # -------------------------------------------------
        # Last Gesture Added To Dashboard History
        # -------------------------------------------------
        self.last_uart_gesture = {

            "Left": None,

            "Right": None,
        }

        print("✅ Video Stream Ready!")

    # =====================================================
    # STATUS
    # =====================================================

    def get_status(self):

        with self.status_lock:

            return {

                "fps": self.status["fps"],

                "cpu": self.status["cpu"],

                "ram": self.status["ram"],

                "hands": self.status["hands"],

                "gesture": self.status["gesture"],

                "confidence": self.status["confidence"],

                "hand": self.status["hand"],

                "command": self.status["command"],

                "packet": self.status["packet"],

                "uart_status": self.status["uart_status"],

                "model_status": self.status["model_status"],

                "camera_status": self.status["camera_status"],
                "aidge_ms": self.status["aidge_ms"],
                "hand_gestures": {
                    "Left": dict(self.status["hand_gestures"]["Left"]),
                    "Right": dict(self.status["hand_gestures"]["Right"]),
                },

                "history": list(
                    self.status["history"]
                ),
            }

    # =====================================================
    # CAMERA RECONNECT
    # =====================================================

    def reconnect_camera(self):

        print("Reconnecting camera...")

        try:
            self.cap.release()

        except Exception:
            pass

        time.sleep(1)

        self.cap = open_camera()

        with self.status_lock:

            self.status[
                "camera_status"
            ] = "CONNECTED"

    # =====================================================
    # FRAME GENERATOR
    # =====================================================

    def generate_frames(self):

        while True:

            # -------------------------------------------------
            # Capture Frame
            # -------------------------------------------------
            success, frame = self.cap.read()

            if not success:

                if (
                    time.time()
                    - self.last_camera_check
                    > 2
                ):

                    self.reconnect_camera()

                    self.last_camera_check = (
                        time.time()
                    )

                continue

            # -------------------------------------------------
            # Mirror Camera
            # -------------------------------------------------
            frame = cv2.flip(
                frame,
                1
            )

            # -------------------------------------------------
            # Performance
            # -------------------------------------------------
            self.performance.update()

            fps = self.performance.get_fps()

            cpu = self.performance.get_cpu()

            ram = self.performance.get_memory()

            # -------------------------------------------------
            # AI Processing
            #
            # Camera
            #   ↓
            # MediaPipe
            #   ↓
            # Normalization
            #   ↓
            # Aidge CNN
            #   ↓
            # UART
            # -------------------------------------------------
            frame, predictions = (
                self.detector.process(
                    frame
                )
            )

            # -------------------------------------------------
            aidge_ms = self.detector.model.last_inference_ms

            current_hands = {

                "Left": {"gesture": "None", "confidence": 0.0},

                "Right": {"gesture": "None", "confidence": 0.0},

            }

            # Current Gesture
            # -------------------------------------------------
            current_gesture = "None"

            current_confidence = 0.0

            current_hand = "None"

            # -------------------------------------------------
            # Process Predictions
            # -------------------------------------------------
            for pred in predictions:

                gesture = pred["gesture"]

                hand = pred["hand"]

                confidence = pred["confidence"]

                if hand in current_hands:
                    current_hands[hand] = {
                        "gesture": gesture,
                        "confidence": confidence,
                    }

                # -------------------------------------------------
                # Ignore Unknown Predictions
                # -------------------------------------------------
                if gesture == "Unknown":
                    continue

                # -------------------------------------------------
                # Current Dashboard Gesture
                # -------------------------------------------------
                current_gesture = gesture

                current_confidence = confidence

                current_hand = hand

                # -------------------------------------------------
                # UART Information From Detector
                # -------------------------------------------------
                uart_info = pred.get(
                    "uart"
                )

                if (
                    uart_info
                    and uart_info.get(
                        "command"
                    ) is not None
                ):

                    with self.status_lock:

                        self.status[
                            "command"
                        ] = uart_info[
                            "command"
                        ]

                        self.status[
                            "packet"
                        ] = uart_info[
                            "packet"
                        ]

                        self.status[
                            "uart_status"
                        ] = uart_info[
                            "uart_status"
                        ]

                # -------------------------------------------------
                # Add Gesture To Dashboard History
                # Only when gesture changes
                # -------------------------------------------------
                if (
                    self.last_uart_gesture.get(
                        hand
                    )
                    != gesture
                ):

                    self.history.add(
                        hand,
                        gesture
                    )

                    self.last_uart_gesture[
                        hand
                    ] = gesture

            # -------------------------------------------------
            # Get History
            # -------------------------------------------------
            history = self.history.get()

            # -------------------------------------------------
            # Camera Overlay
            # -------------------------------------------------
            frame = self.overlay.draw(

                frame,

                predictions,

                fps,

                cpu,

                ram,

                history,
            )

            # -------------------------------------------------
            # Update Dashboard Status
            # -------------------------------------------------
            with self.status_lock:

                self.status[
                    "fps"
                ] = fps

                self.status[
                    "cpu"
                ] = cpu

                self.status[
                    "ram"
                ] = ram

                self.status[
                    "hands"
                ] = len(predictions)

                self.status[
                    "gesture"
                ] = current_gesture

                self.status[
                    "confidence"
                ] = current_confidence

                self.status[
                    "hand"
                ] = current_hand

                self.status[
                    "history"
                ] = history

                self.status["aidge_ms"] = aidge_ms
                self.status["hand_gestures"] = current_hands

            # -------------------------------------------------
            # JPEG Encoding
            # -------------------------------------------------
            success, buffer = cv2.imencode(

                ".jpg",

                frame
            )

            if not success:
                continue

            frame_bytes = buffer.tobytes()

            # -------------------------------------------------
            # MJPEG Frame
            # -------------------------------------------------
            yield (

                b"--frame\r\n"

                b"Content-Type: image/jpeg\r\n\r\n"

                + frame_bytes

                + b"\r\n"
            )

    # =====================================================
    # RELEASE
    # =====================================================

    def release(self):

        # -------------------------------------------------
        # Release Camera
        # -------------------------------------------------
        try:

            self.cap.release()

        except Exception:
            pass

        # -------------------------------------------------
        # Close Detector
        #
        # Detector closes the ONLY UART connection.
        # -------------------------------------------------
        try:

            self.detector.close()

        except Exception:
            pass

        # -------------------------------------------------
        # OpenCV Cleanup
        # -------------------------------------------------
        try:

            cv2.destroyAllWindows()

        except Exception:
            pass
