import cv2
import mediapipe as mp


class HandDetector:

    def __init__(
        self,
        max_num_hands=2,
        detection_confidence=0.7,
        tracking_confidence=0.7,
    ):

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

    def detect(self, frame, draw=True):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.hands.process(rgb)

        detected_hands = []

        if results.multi_hand_landmarks:

            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):

                if draw:
                    self.mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                    )

                # Hand type (Left / Right)
                hand_type = "Unknown"

                if results.multi_handedness:
                    hand_type = results.multi_handedness[idx].classification[0].label

                # Bounding box
                x_list = []
                y_list = []

                for lm in hand_landmarks.landmark:
                    x_list.append(lm.x)
                    y_list.append(lm.y)

                x_min = min(x_list)
                x_max = max(x_list)
                y_min = min(y_list)
                y_max = max(y_list)

                detected_hands.append({
                    "id": idx,
                    "type": hand_type,
                    "landmarks": hand_landmarks,
                    "bbox": (
                        x_min,
                        y_min,
                        x_max,
                        y_max,
                    ),
                })

        return frame, detected_hands