"""
UART sender for transmitting hexadecimal packets.
"""

import serial
from communication.packet import create_packet


class UARTSender:
    def __init__(self, port="/dev/serial0", baudrate=115200):
        try:
            self.serial = serial.Serial(
                port=port,
                baudrate=baudrate,
                timeout=1
            )
            print(f"✅ UART initialized on {port}")

        except Exception as e:
            self.serial = None
            print(f"❌ UART initialization failed: {e}")

    def send(self, command: int):
        if self.serial is None:
            print("⚠ UART not available")
            return

        packet = create_packet(command)

        try:
            self.serial.write(packet)

            print("\n==============================")
            print(" UART TRANSMISSION")
            print("==============================")
            print(f"Packet   : {packet.hex().upper()}")
            print(f"Command  : 0x{command:02X}")
            print("Status   : SUCCESS")
            print("==============================\n")

        except Exception as e:
            print(f"UART Send Error: {e}")

    def close(self):
        if self.serial:
            self.serial.close()
