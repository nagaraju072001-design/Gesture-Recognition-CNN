import cv2
from pathlib import Path
from datetime import datetime


class VideoRecorder:

    def __init__(self, output_dir="recordings", fps=20):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.writer = None
        self.recording = False
        self.fps = fps

    def start(self, frame):
        if self.recording:
            return

        h, w = frame.shape[:2]

        filename = datetime.now().strftime("gesture_%Y%m%d_%H%M%S.mp4")
        filepath = self.output_dir / filename

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        self.writer = cv2.VideoWriter(
            str(filepath),
            fourcc,
            self.fps,
            (w, h),
        )

        self.recording = True

    def write(self, frame):
        if self.recording and self.writer:
            self.writer.write(frame)

    def stop(self):
        if self.writer:
            self.writer.release()

        self.writer = None
        self.recording = False

    def is_recording(self):
        return self.recording
