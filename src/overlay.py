import cv2


class Overlay:

    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def draw(self, frame, predictions, fps, cpu, ram, history):

        h, w = frame.shape[:2]

        GREEN = (0, 230, 100)
        WHITE = (245, 245, 245)
        DARK = (25, 25, 25)

        # -------------------------------------------------
        # Camera border
        # -------------------------------------------------

        cv2.rectangle(
            frame,
            (5, 5),
            (w - 5, h - 5),
            GREEN,
            2,
        )

        # -------------------------------------------------
        # Hand detection
        # -------------------------------------------------

        for pred in predictions:

            x1, y1, x2, y2 = pred["bbox"]

            x1 = int(x1 * w)
            y1 = int(y1 * h)
            x2 = int(x2 * w)
            y2 = int(y2 * h)

            color = GREEN

            # Bounding box

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2,
            )

            # -------------------------------------------------
            # Gesture label
            # -------------------------------------------------

            label = (
                f"{pred['hand']}   "
                f"{pred['gesture']}   "
                f"{pred['confidence']:.1f}%"
            )

            text_size = cv2.getTextSize(
                label,
                self.font,
                0.55,
                1,
            )[0]

            label_width = text_size[0] + 20
            label_height = 35

            label_x = x1
            label_y = max(
                5,
                y1 - label_height
            )

            cv2.rectangle(
                frame,
                (
                    label_x,
                    label_y,
                ),
                (
                    label_x + label_width,
                    label_y + label_height,
                ),
                GREEN,
                -1,
            )

            cv2.putText(
                frame,
                label,
                (
                    label_x + 10,
                    label_y + 23,
                ),
                self.font,
                0.55,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

            # -------------------------------------------------
            # Confidence bar
            # -------------------------------------------------

            bar_width = 150
            bar_height = 8

            bar_x = x1
            bar_y = min(
                h - 15,
                y2 + 12
            )

            filled = int(
                bar_width *
                pred["confidence"] /
                100
            )

            cv2.rectangle(
                frame,
                (
                    bar_x,
                    bar_y,
                ),
                (
                    bar_x + bar_width,
                    bar_y + bar_height,
                ),
                (70, 70, 70),
                -1,
            )

            cv2.rectangle(
                frame,
                (
                    bar_x,
                    bar_y,
                ),
                (
                    bar_x + filled,
                    bar_y + bar_height,
                ),
                GREEN,
                -1,
            )

        return frame
