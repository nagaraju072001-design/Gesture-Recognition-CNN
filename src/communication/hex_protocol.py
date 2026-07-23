"""
Hexadecimal command definitions for gesture recognition.
"""

from enum import IntEnum


class HexCommand(IntEnum):
    START = 0xA1
    STOP = 0xA2
    NEXT = 0xA3
    ACCEPT = 0xA4
    CONFIRM = 0xA5


GESTURE_MAP = {
    "Palm": HexCommand.START,
    "Fist": HexCommand.STOP,
    "Peace": HexCommand.NEXT,
    "Thumb_Up": HexCommand.ACCEPT,
    "OK": HexCommand.CONFIRM,
}


def get_command(gesture: str):
    """
    Return the hexadecimal command for the detected gesture.
    """
    return GESTURE_MAP.get(gesture)
