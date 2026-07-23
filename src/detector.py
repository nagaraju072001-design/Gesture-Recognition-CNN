from collections import Counter, deque

from hand_detector import HandDetector
from model import GestureModel
from normalize import LandmarkNormalizer

# UART Communication
from communication.hex_protocol import get_command
from communication.uart_sender import UARTSender


class GestureDetector:

    def __init__(
        self,
        confidence_threshold=0.90,
        history_size=10,
    ):

        print("Initializing Gesture Detector...")

        self.hand_detector = HandDetector(
            max_num_hands=2,
            detection_confidence=0.7,
            tracking_confidence=0.7,
        )

        self.model = GestureModel()

        self.normalizer = LandmarkNormalizer()

        # UART Sender
        self.uart = UARTSender()

        # Confidence threshold
        self.confidence_threshold = confidence_threshold

        # Gesture history
        self.history = {
            "Left": deque(maxlen=history_size),
            "Right": deque(maxlen=history_size),
        }

        # Last transmitted gesture
        self.last_sent = {
            "Left": None,
            "Right": None,
        }

        print("✅ Gesture Detector Ready!")

    def process(self, frame):

        frame, hands = self.hand_detector.detect(frame)

        predictions = []

        for hand in hands:

            landmarks = []

            # ---------------------------------------
            # Extract landmarks
            # ---------------------------------------
            for lm in hand["landmarks"].landmark:

                landmarks.extend([
                    lm.x,
                    lm.y,
                    lm.z,
                ])

            # ---------------------------------------
            # Normalize landmarks
            # ---------------------------------------
            landmarks = self.normalizer.normalize(landmarks)

            # ---------------------------------------
            # AI Prediction
            # ---------------------------------------
            gesture, confidence = self.model.predict(landmarks)

            hand_type = hand["type"]

            # ---------------------------------------
            # Confidence filtering
            # ---------------------------------------
            if confidence < self.confidence_threshold:

                predictions.append({

                    "hand": hand_type,

                    "gesture": "Unknown",

                    "confidence": round(confidence * 100, 1),

                    "bbox": hand["bbox"],

                    "landmarks": hand["landmarks"]

                })

                continue

            # ---------------------------------------
            # Prediction History
            # ---------------------------------------
            self.history[hand_type].append(gesture)

            stable_gesture = Counter(
                self.history[hand_type]
            ).most_common(1)[0][0]

            # ---------------------------------------
            # UART Transmission
            # ---------------------------------------
            if stable_gesture != self.last_sent[hand_type]:

                command = get_command(stable_gesture)

                if command is not None:
                    self.uart.send(command)

                self.last_sent[hand_type] = stable_gesture

            # ---------------------------------------
            # Output
            # ---------------------------------------
            predictions.append({

                "hand": hand_type,

                "gesture": stable_gesture,

                "confidence": round(confidence * 100, 1),

                "bbox": hand["bbox"],

                "landmarks": hand["landmarks"]

            })

        return frame, predictions

    def close(self):
        """Close UART when application exits."""
        self.uart.close()
