# DriveGuard-AI 🚗💨👁️

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-Backend-green.svg)](https://flask.palletsprojects.com/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-orange.svg)](https://docs.ultralytics.com/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Face%20Mesh-red.svg)](https://developers.google.com/mediapipe)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**DriveGuard-AI** is a real-time, AI-powered driver safety and monitoring system designed to prevent road accidents caused by driver fatigue, distraction, and unsafe driving behaviors. By leveraging Computer Vision, MediaPipe Facial Landmarks, and fine-tuned YOLOv8 neural networks, DriveGuard-AI continuously analyzes live video streams to deliver instant visual, audio, and web dashboard alerts.

---

## 🌟 Key Features

- 👁️ **Drowsiness & Eye Closure Detection**: Computes Eye Aspect Ratio (EAR) using 468 MediaPipe facial landmarks to identify prolonged eye closure and micro-sleeps.
- 🥱 **Yawning & Fatigue Analysis**: Tracks Mouth Aspect Ratio (MAR) to detect repeated yawning and exhaustion.
- 🗣️ **Head Pose & Distraction Tracking**: Monitors pitch, yaw, and roll to detect when the driver takes their eyes off the road.
- 📱 **Mobile Phone Detection**: Uses custom-trained YOLOv8 object detection to flag phone usage while driving.
- 🚬 **Smoking & Cigarette Detection**: Identifies smoking activity inside the cabin to maintain vehicle safety standards.
- 🦺 **Seatbelt Compliance Monitoring**: Verifies if the driver is properly wearing a seatbelt.
- 🕶️ **Sunglasses Handling**: Adapts facial tracking dynamically when sunglasses are detected.
- 🌙 **Low-Light Enhancement**: Pre-processes low-contrast / nighttime camera frames for reliable night driving safety.
- 🔊 **Multi-Modal Audio & Speech Alerts**: Triggers real-time audio alarms (`playsound`) and text-to-speech warnings (`pyttsx3`).
- 📊 **Real-Time Web Dashboard**: Flask-powered interactive web dashboard for real-time live telemetry, alert history, and driver login.

---

## 🛠️ Tech Stack & Architecture

- **Core Language**: Python 3.10+
- **Computer Vision**: OpenCV (`cv2`)
- **Facial Landmark Mesh**: Google MediaPipe Face Mesh
- **Object Detection**: Ultralytics YOLOv8 (Custom-trained weights `best.pt`)
- **Backend & Web API**: Flask, SQLite
- **Frontend**: HTML5, CSS3 (Glassmorphism design), Modern JavaScript
- **Audio System**: `playsound`, `pyttsx3` (TTS)

```
DriveGuard-AI/
├── backend/                  # Flask REST API & Web Dashboard
│   ├── app.py                # Main Flask web application
│   ├── database.py           # SQLite event & driver database
│   ├── templates/            # Web interface (Dashboard, Login)
│   └── static/               # CSS styles, JS scripts, live telemetry JSON
├── detection/                # Computer Vision & Deep Learning Engine
│   ├── detect.py             # Core real-time detection pipeline loop
│   ├── eye_detection.py      # Eye Aspect Ratio (EAR) calculator
│   ├── mouth_detection.py    # Yawn detection (MAR)
│   ├── head_pose.py          # 3D Head pose estimation
│   ├── phone_detector.py     # Cell phone usage detection
│   ├── smoking_detector.py   # Smoking detection
│   ├── seatbelt_detector.py  # Seatbelt verification
│   ├── alarm.py              # Audio alert & siren dispatcher
│   └── notifier.py           # Desktop and web notification engine
├── driver_custom_model/      # Fine-tuned YOLOv8 model & training results
│   └── yolov8n_custom_v1/    # Weights (`best.pt`) and evaluation metrics
├── datasets/                 # Custom annotated datasets (YOLO format)
├── run_system.ps1            # One-click startup script for Windows
└── requirements.txt          # Python dependencies
```

---

## 🚀 Quick Start & Installation

### 1. Prerequisites
- **Python 3.10 or higher**
- Webcam / Camera input device

### 2. Clone the Repository
```bash
git clone https://github.com/Ganesh-gouli/DriveGuard-AI.git
cd DriveGuard-AI
```

### 3. Set Up Virtual Environment & Install Dependencies
```bash
python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

---

## 💻 Running DriveGuard-AI

### Option A: One-Click Startup (Windows PowerShell)
Run the automated startup script:
```powershell
.\run_system.ps1
```

### Option B: Manual Startup

1. **Start the Flask Web Dashboard**:
   ```bash
   python backend/app.py
   ```
   *Dashboard will be available at: `http://localhost:5000`*

2. **Start the Computer Vision Engine** (in a separate terminal):
   ```bash
   python detection/detect.py
   ```

---

## 🎯 Custom Model Training

To fine-tune the YOLOv8 object detector on custom datasets for phone usage or smoking detection:

```bash
# Label & prepare custom datasets
python auto_label.py

# Train custom YOLOv8 model
python train_custom.py
```

Evaluation metrics, confusion matrices, and model weights will be saved to `driver_custom_model/`.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more details.

---

## 👨‍💻 Author

Developed with ❤️ by **[Ganesh Gouli](https://github.com/Ganesh-gouli)**.
