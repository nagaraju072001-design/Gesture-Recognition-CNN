# AI-Based Hand Gesture Recognition using CNN, MediaPipe, Raspberry Pi 5 & UART Communication

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Tracking-red)
![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-5-darkred)
![License](https://img.shields.io/badge/License-MIT-blue)

---

## 📌 Project Overview

This project implements a **real-time AI-based hand gesture recognition system** using **MediaPipe** and a **Convolutional Neural Network (CNN)** running on a **Raspberry Pi 5**.

The system detects hand gestures through a USB camera, classifies them using a trained CNN model, and transmits hexadecimal UART commands to an embedded device for real-time control.

This project demonstrates the integration of **Computer Vision**, **Deep Learning**, and **Embedded Systems**.

---

## 🎯 Features

- Real-time hand detection using MediaPipe
- CNN-based gesture classification
- Raspberry Pi 5 implementation
- USB Camera support
- UART communication
- Hexadecimal packet protocol
- Stable gesture detection
- Modular Python architecture
- Ready for embedded system integration

---

## 🛠 Hardware Requirements

- Raspberry Pi 5
- Logitech C270 USB Camera (or compatible USB camera)
- MicroSD Card
- UART-enabled Embedded Board (Optional)
- USB Keyboard & Monitor (for setup)

---

## 💻 Software Requirements

- Raspberry Pi OS
- Python 3.11+
- TensorFlow
- OpenCV
- MediaPipe
- NumPy
- PySerial
- Flask

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## 📂 Project Structure

```
Gesture-Recognition-CNN/
│
├── models/
│   └── gesture_model.keras
│
├── screenshots/
│
├── src/
│   ├── camera.py
│   ├── detector.py
│   ├── predictor.py
│   ├── stream.py
│   ├── web_stream.py
│   └── communication/
│       ├── hex_protocol.py
│       ├── packet.py
│       └── uart_sender.py
│
├── templates/
│   └── index.html
│
├── uart_test.py
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🖐 Supported Gestures

| Gesture | UART Command |
|----------|--------------|
| Palm | 0xA1 |
| Fist | 0xA2 |
| Peace | 0xA3 |
| Thumb Up | 0xA4 |
| OK | 0xA5 |

---

## 📡 UART Packet Format

Each transmitted packet contains three bytes.

| Byte | Description |
|------|-------------|
| Byte 1 | Start Byte (0xAA) |
| Byte 2 | Gesture Command |
| Byte 3 | Checksum (0xAA XOR Command) |

### Example

Palm Gesture

```
AA A1 0B
```

Thumb Up

```
AA A4 0E
```

---

## 🔄 System Workflow

```
USB Camera
      │
      ▼
MediaPipe Hand Detection
      │
      ▼
Landmark Extraction
      │
      ▼
CNN Gesture Classification
      │
      ▼
Stable Gesture Verification
      │
      ▼
Hexadecimal Command Generation
      │
      ▼
UART Transmission
      │
      ▼
Embedded System
```

---

## ▶ Running the Project

Clone the repository

```bash
git clone git@github.com:nagaraju072001-design/Gesture-Recognition-CNN.git
```

Move into the project

```bash
cd Gesture-Recognition-CNN
```

Install dependencies

```bash
pip install -r requirements.txt
```

Start the application

```bash
python app.py
```

or

```bash
python uart_test.py
```

---

## 📸 Demo

Add screenshots inside the **screenshots/** folder.

Example:

```
screenshots/
    live_detection.png
    prediction.png
    uart_output.png
```

You can display them here later.

---

## 🚀 Future Improvements

- Dynamic gesture recognition
- LSTM-based sequence model
- Bluetooth communication
- Wi-Fi communication
- MQTT support
- Mobile application integration
- Edge AI optimization

---

## 📚 Technologies Used

- Python
- TensorFlow
- OpenCV
- MediaPipe
- NumPy
- Flask
- PySerial
- Raspberry Pi 5

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Nagaraju**

GitHub:
https://github.com/nagaraju072001-design

---

## ⭐ If you found this project useful, consider giving it a Star!
