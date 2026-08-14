import time
import threading
import cv2

from camera import open_camera
from detector import GestureDetector
from overlay import Overlay
from performance import PerformanceMonitor
from history import GestureHistory
from recorder import VideoRecorder

from communication.hex_protocol import get_command
from communication.packet import create_packet
from communication.uart_sender import UARTSender


class VideoStream:

    def __init__(self):

        print("Starting Video Stream...")

        self.cap = open_camera()

        self.detector = GestureDetector()

        self.overlay = Overlay()

        self.performance = PerformanceMonitor()

        self.history = GestureHistory()

        self.recorder = VideoRecorder()

        self.last_camera_check = time.time()

        # -------------------------------------------------
        # UART
        # -------------------------------------------------

        self.uart = UARTSender()

        # -------------------------------------------------
        # Latest dashboard information
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

            "history": [],
        }

        # Prevent repeated UART transmissions

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

                "history": list(self.status["history"]),
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
            self.status["camera_status"] = "CONNECTED"

    # =====================================================
    # UART
    # =====================================================

    def send_uart(self, gesture):

        try:

            command = get_command(gesture)

            if command is None:
                return

            command_value = int(command)

            packet = create_packet(command_value)

            self.uart.send(command_value)

            with self.status_lock:

                self.status["command"] = (
                    f"0x{command_value:02X}"
                )

                self.status["packet"] = (
                    packet.hex().upper()
                )

                self.status["uart_status"] = (
                    "SENT SUCCESSFULLY"
                )

        except Exception as e:

            print(f"UART Error: {e}")

            with self.status_lock:
                self.status["uart_status"] = "ERROR"

    # =====================================================
    # FRAME GENERATOR
    # =====================================================

    def generate_frames(self):

        while True:

            success, frame = self.cap.read()

            if not success:

                if (
                    time.time() -
                    self.last_camera_check
                    > 2
                ):

                    self.reconnect_camera()

                    self.last_camera_check = time.time()

                continue

            # Mirror camera

            frame = cv2.flip(frame, 1)

            # -------------------------------------------------
            # Performance
            # -------------------------------------------------

            self.performance.update()

            fps = self.performance.get_fps()

            cpu = self.performance.get_cpu()

            ram = self.performance.get_memory()

            # -------------------------------------------------
            # AI
            # -------------------------------------------------

            frame, predictions = self.detector.process(
                frame
            )

            # -------------------------------------------------
            # Gesture handling
            # -------------------------------------------------

            current_gesture = "None"
            current_confidence = 0.0
            current_hand = "None"

            for pred in predictions:

                gesture = pred["gesture"]

                hand = pred["hand"]

                confidence = pred["confidence"]

                if gesture == "Unknown":
                    continue

                current_gesture = gesture

                current_confidence = confidence

                current_hand = hand

                # ---------------------------------------------
                # Add history only when gesture changes
                # ---------------------------------------------

                if (
                    self.last_uart_gesture.get(hand)
                    != gesture
                ):

                    self.history.add(
                        hand,
                        gesture
                    )

                    self.last_uart_gesture[hand] = gesture

                    # UART

                    self.send_uart(gesture)

            # -------------------------------------------------
            # History
            # -------------------------------------------------

            history = self.history.get()

            # -------------------------------------------------
            # Camera overlay
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
            # Update dashboard status
            # -------------------------------------------------

            with self.status_lock:

                self.status["fps"] = fps

                self.status["cpu"] = cpu

                self.status["ram"] = ram

                self.status["hands"] = len(predictions)

                self.status["gesture"] = current_gesture

                self.status["confidence"] = (
                    current_confidence
                )

                self.status["hand"] = current_hand

                self.status["history"] = history

            # -------------------------------------------------
            # JPEG
            # -------------------------------------------------

            success, buffer = cv2.imencode(
                ".jpg",
                frame
            )

            if not success:
                continue

            frame_bytes = buffer.tobytes()

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

        try:
            self.cap.release()
        except Exception:
            pass

        try:
            self.uart.close()
        except Exception:
            pass

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
