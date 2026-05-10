# SignEase — Unified Sign Language Detection

A premium real-time **Sign Language Recognition System** that supports both **American Sign Language (ASL)** and **Indian Sign Language (ISL)** in a single Flask application. Switch between modes instantly with live webcam detection, sentence building, and text-to-speech.

---

## 🚀 Key Features

| Feature | ASL Mode | ISL Mode |
|---------|----------|----------|
| **Model Type** | Random Forest (Scikit-learn) | Deep Neural Network (TensorFlow) |
| **Hand Detection** | MediaPipe Tasks API | MediaPipe Hands (Solutions) |
| **Hands Supported** | 1 hand | 2 hands (84 keypoints) |
| **Classes** | A-Z + Space (26) | 1-9, A-Z (35) |
| **Special Gestures** | — | `1` = Space, `2` = Delete Word |

### Shared Features
- ✅ **Instant Mode Switching** — Toggle between ASL and ISL without restart
- ✅ **Real-Time Sentence Builder** — Accumulates detected signs into text
- ✅ **Text-to-Speech (TTS)** — Speak the accumulated sentence aloud
- ✅ **Prediction Stability** — Cooldown and buffer mechanisms prevent flickering
- ✅ **Premium Dark UI** — Glassmorphism theme with micro-animations
- ✅ **Responsive Design** — Works on desktop, tablet, and mobile
- ✅ **System Status Dashboard** — Live model and TTS status indicators

---

## 🧠 Technical Architecture

```text
Unified_Sign_Language/
│
├── app.py                      # Flask backend (dual-mode detection)
├── templates/
│   └── index.html              # Premium dark-themed UI
├── static/
│   ├── style.css               # Glassmorphism CSS design system
│   ├── script.js               # Frontend interactions & polling
│   └── images/
│       ├── asl_signs.jpeg      # ASL gesture reference chart
│       └── isl_gestures.png    # ISL gesture reference chart
│
├── models/
│   ├── asl_model.p             # ASL Random Forest model (pickle)
│   ├── hand_landmarker.task    # MediaPipe hand landmarker model
│   ├── isl_model.h5            # ISL TensorFlow/Keras DNN model
│   └── isl_label_classes.json  # ISL label mapping
│
├── generate_keypoints.py       # ISL data extraction pipeline
├── train_model.py              # ISL model training pipeline
├── environment.yml             # Conda environment specification
├── requirements.txt            # Pip dependencies
└── README.md                   # This file
```

---

## 🛠 Getting Started

### Prerequisites
- **Python 3.10** (required for TensorFlow + MediaPipe compatibility)
- **Webcam** (built-in or external)
- **Conda** (recommended) or virtualenv

### 1. Create Conda Environment
```bash
conda env create -f environment.yml
conda activate sign_language_unified
```

### 2. Or use pip
```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 3. Run the Application
```bash
python app.py
```
Open **http://localhost:5000** in your browser.

---

## 📈 Retraining the ISL Model

1. Organize images in `dataset from kaggle/` directory
2. Extract keypoints:
   ```bash
   python generate_keypoints.py
   ```
3. Train model:
   ```bash
   python train_model.py
   ```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Main UI page |
| GET | `/video_feed` | MJPEG video stream |
| GET | `/get_data` | Current prediction + sentence |
| GET | `/model_status` | Model & TTS status |
| POST | `/switch_mode` | Switch ASL/ISL (`{"mode": "ISL"}`) |
| POST | `/clear_sentence` | Clear accumulated text |
| POST | `/speak` | Speak accumulated text via TTS |
| POST | `/stop` | Stop video stream |
| POST | `/delete_last` | Delete last character |
| POST | `/add_space` | Add space to sentence |

---

## 🤝 Credits

**Author**: Nachiket Shinde — KodeNeurons  
**AI Assistant**: Antigravity AI

---
⭐ **If you find this project helpful, please give it a star!**
