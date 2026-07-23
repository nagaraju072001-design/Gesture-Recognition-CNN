import cv2


class Overlay:

    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def draw(
        self,
        frame,
        predictions,
        fps,
        cpu,
        ram,
        history,
    ):

        h, w = frame.shape[:2]

        # -----------------------------
        # HEADER
        # -----------------------------

        cv2.rectangle(frame, (0, 0), (w, 90), (35, 35, 35), -1)

        cv2.putText(
            frame,
            "AI Hand Gesture Recognition",
            (20, 30),
            self.font,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"FPS : {fps:.1f}",
            (20, 65),
            self.font,
            0.6,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"CPU : {cpu:.1f}%",
            (150, 65),
            self.font,
            0.6,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"RAM : {ram:.1f}%",
            (300, 65),
            self.font,
            0.6,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Hands : {len(predictions)}",
            (450, 65),
            self.font,
            0.6,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            "Status : RUNNING",
            (600, 65),
            self.font,
            0.6,
            (0, 255, 0),
            2,
        )

        # -----------------------------
        # HANDS
        # -----------------------------

        for pred in predictions:

            x1, y1, x2, y2 = pred["bbox"]

            x1 = int(x1 * w)
            y1 = int(y1 * h)
            x2 = int(x2 * w)
            y2 = int(y2 * h)

            color = (0, 255, 0) if pred["hand"] == "Right" else (255, 120, 0)

            cv2.rectangle(
                frame,
                (x1 - 15, y1 - 15),
                (x2 + 15, y2 + 15),
                color,
                2,
            )

            cv2.rectangle(
                frame,
                (x1 - 15, y1 - 70),
                (x1 + 200, y1 - 15),
                color,
                -1,
            )

            cv2.putText(
                frame,
                pred["hand"],
                (x1, y1 - 48),
                self.font,
                0.55,
                (0, 0, 0),
                2,
            )

            cv2.putText(
                frame,
                pred["gesture"],
                (x1, y1 - 25),
                self.font,
                0.65,
                (0, 0, 0),
                2,
            )

            cv2.putText(
                frame,
                f"{pred['confidence']:.1f}%",
                (x1 + 95, y1 - 25),
                self.font,
                0.55,
                (0, 0, 0),
                2,
            )

            # -----------------------------
            # Confidence Bar
            # -----------------------------

            bar_width = 120

            filled = int(bar_width * pred["confidence"] / 100)

            cv2.rectangle(
                frame,
                (x1, y2 + 15),
                (x1 + bar_width, y2 + 30),
                (80, 80, 80),
                -1,
            )

            cv2.rectangle(
                frame,
                (x1, y2 + 15),
                (x1 + filled, y2 + 30),
                color,
                -1,
            )

        # -----------------------------
        # Gesture History
        # -----------------------------

        panel_x = w - 260

        cv2.rectangle(
            frame,
            (panel_x, 100),
            (w - 10, 300),
            (40, 40, 40),
            -1,
        )

        cv2.putText(
            frame,
            "Recent Gestures",
            (panel_x + 10, 125),
            self.font,
            0.6,
            (0, 255, 255),
            2,
        )

        y = 155

        for item in history[:6]:

            cv2.putText(
                frame,
                f"{item['hand']} : {item['gesture']}",
                (panel_x + 10, y),
                self.font,
                0.5,
                (255, 255, 255),
                1,
            )

            y += 25

        return frame
