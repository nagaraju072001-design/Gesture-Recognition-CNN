from collections import Counter, deque

from hand_detector import HandDetector
from aidge_model import AidgeGestureModel
from normalize import LandmarkNormalizer

# UART Communication
from communication.hex_protocol import get_command
from communication.packet import create_packet
from communication.uart_sender import UARTSender


class GestureDetector:
    def __init__(
        self,
        confidence_threshold=0.90,
        history_size=10,
    ):
        print("Initializing Gesture Detector...")

        # -------------------------------------------------
        # Hand Detection
        # -------------------------------------------------
        self.hand_detector = HandDetector(
            max_num_hands=2,
            detection_confidence=0.7,
            tracking_confidence=0.7,
        )

        # -------------------------------------------------
        # Aidge AI Model
        # -------------------------------------------------
        self.model = AidgeGestureModel()

        # -------------------------------------------------
        # Landmark Normalization
        # -------------------------------------------------
        self.normalizer = LandmarkNormalizer()

        # -------------------------------------------------
        # UART
        # -------------------------------------------------
        self.uart = UARTSender()

        # -------------------------------------------------
        # Configuration
        # -------------------------------------------------
        self.confidence_threshold = confidence_threshold

        # -------------------------------------------------
        # Gesture History
        # -------------------------------------------------
        self.history = {
            "Left": deque(maxlen=history_size),
            "Right": deque(maxlen=history_size),
        }

        # -------------------------------------------------
        # Last Transmitted Gesture
        # -------------------------------------------------
        self.last_sent = {
            "Left": None,
            "Right": None,
        }

        print("✅ Gesture Detector Ready!")

    # =====================================================
    # PROCESS FRAME
    # =====================================================

    def process(self, frame):

        # -------------------------------------------------
        # Detect Hands
        # -------------------------------------------------
        frame, hands = self.hand_detector.detect(frame)

        predictions = []

        # -------------------------------------------------
        # Process Each Hand
        # -------------------------------------------------
        for hand in hands:

            # -------------------------------------------------
            # Extract 21 Hand Landmarks
            # -------------------------------------------------
            landmarks = []

            for lm in hand["landmarks"].landmark:
                landmarks.extend([
                    lm.x,
                    lm.y,
                    lm.z,
                ])

            # -------------------------------------------------
            # Normalize Landmarks
            # -------------------------------------------------
            landmarks = self.normalizer.normalize(landmarks)

            # -------------------------------------------------
            # Aidge AI Prediction
            # -------------------------------------------------
            gesture, confidence = self.model.predict(
                landmarks
            )

            hand_type = hand["type"]

            # -------------------------------------------------
            # Confidence Filtering
            # -------------------------------------------------
            if confidence < self.confidence_threshold:

                predictions.append({
                    "hand": hand_type,
                    "gesture": "Unknown",
                    "confidence": round(
                        confidence * 100,
                        1
                    ),
                    "bbox": hand["bbox"],
                    "landmarks": hand["landmarks"],
                    "uart": {
                        "command": None,
                        "packet": None,
                        "uart_status": "READY",
                    },
                })

                continue

            # -------------------------------------------------
            # Gesture History
            # -------------------------------------------------
            self.history[hand_type].append(
                gesture
            )

            stable_gesture = Counter(
                self.history[hand_type]
            ).most_common(1)[0][0]

            # -------------------------------------------------
            # UART Information
            # -------------------------------------------------
            uart_info = {
                "command": None,
                "packet": None,
                "uart_status": "READY",
            }

            # -------------------------------------------------
            # UART Transmission
            # Only transmit when gesture changes
            # -------------------------------------------------
            if (
                stable_gesture
                != self.last_sent[hand_type]
            ):

                command = get_command(
                    stable_gesture
                )

                if command is not None:

                    try:

                        command_value = int(command)

                        # Send command through the
                        # single UART instance
                        self.uart.send(
                            command_value
                        )

                        # Create packet for dashboard
                        packet = create_packet(
                            command_value
                        )

                        uart_info = {
                            "command": (
                                f"0x{command_value:02X}"
                            ),
                            "packet": (
                                packet.hex().upper()
                            ),
                            "uart_status": (
                                "SENT SUCCESSFULLY"
                            ),
                        }

                    except Exception as e:

                        print(
                            f"UART Error: {e}"
                        )

                        uart_info = {
                            "command": (
                                f"0x{int(command):02X}"
                            ),
                            "packet": "--",
                            "uart_status": "ERROR",
                        }

                self.last_sent[hand_type] = (
                    stable_gesture
                )

            # -------------------------------------------------
            # Prediction Output
            # -------------------------------------------------
            predictions.append({

                "hand": hand_type,

                "gesture": stable_gesture,

                "confidence": round(
                    confidence * 100,
                    1
                ),

                "bbox": hand["bbox"],

                "landmarks": hand["landmarks"],

                # UART information is passed
                # to VideoStream/dashboard
                "uart": uart_info,
            })

        return frame, predictions

    # =====================================================
    # CLOSE
    # =====================================================

    def close(self):

        """Close UART when application exits."""

        try:
            self.uart.close()

        except Exception as e:
            print(
                f"UART close error: {e}"
            )
