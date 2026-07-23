from collections import deque


class GestureHistory:

    def __init__(self, max_history=10):

        self.max_history = max_history

        self.history = deque(maxlen=max_history)

    def add(self, hand, gesture):

        self.history.appendleft({
            "hand": hand,
            "gesture": gesture
        })

    def get(self):

        return list(self.history)

    def clear(self):

        self.history.clear()
