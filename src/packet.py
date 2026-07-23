"""
Packet creation for UART communication.

Packet Format:
------------------------------------
Byte 0 : Start Byte (0xAA)
Byte 1 : Command
Byte 2 : Checksum (Start Byte XOR Command)
------------------------------------
"""

START_BYTE = 0xAA


def calculate_checksum(command: int) -> int:
    """Calculate packet checksum."""
    return START_BYTE ^ command


def create_packet(command: int) -> bytes:
    """Create a UART packet."""
    checksum = calculate_checksum(command)

    return bytes([
        START_BYTE,
        command,
        checksum
    ])
