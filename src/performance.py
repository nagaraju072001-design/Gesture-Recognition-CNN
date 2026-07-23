import time
import psutil


class PerformanceMonitor:

    def __init__(self):

        self.previous = time.time()

        self.fps = 0

    def update(self):

        current = time.time()

        dt = current - self.previous

        self.previous = current

        if dt > 0:
            self.fps = 1.0 / dt

    def get_fps(self):

        return round(self.fps, 1)

    def get_cpu(self):

        return psutil.cpu_percent(interval=None)

    def get_memory(self):

        return psutil.virtual_memory().percent
