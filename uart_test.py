import serial


try:
    ser = serial.Serial("/dev/serial0", 115200, timeout=1)

    print("✅ UART is working!")
    print(f"Connected to: {ser.port}")
    print(f"Baud Rate: {ser.baudrate}")

    ser.close()

except Exception as e:
    print("❌ UART Error:", e)
