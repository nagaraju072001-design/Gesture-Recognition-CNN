import cv2
from pathlib import Path
from datetime import datetime


class ScreenshotManager:

    def __init__(self):
        self.output_dir = Path("screenshots")
        self.output_dir.mkdir(exist_ok=True)

    def save(self, frame):

        filename = datetime.now().strftime(
            "gesture_%Y%m%d_%H%M%S.jpg"
        )

        path = self.output_dir / filename

        cv2.imwrite(str(path), frame)

        print(f"📸 Screenshot saved: {path}")
