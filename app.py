"""
Unified Sign Language Detection - Flask Application
Supports both ASL (American Sign Language) and ISL (Indian Sign Language)
Uses MediaPipe Tasks API (HandLandmarker) for both modes.
Author: Nikhil Pawase -- KodeNeurons
"""

from flask import Flask, render_template, Response, request, jsonify, session, redirect, url_for, flash
from flask_cors import CORS
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()  # Load variables from .env file automatically
import bcrypt
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime
from functools import wraps
import cv2
import copy
import itertools
import pickle
import json
import string
import time
import numpy as np
import warnings
import threading
import subprocess
import sys

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# TensorFlow / Keras import
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tf_keras as keras

# MediaPipe Tasks API imports (works with mediapipe >= 0.10.x)
from mediapipe.tasks.python.vision import hand_landmarker
from mediapipe.tasks.python.core import base_options
from mediapipe import Image as MpImage, ImageFormat

import pyttsx3

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'signease-fallback-secret-key')
CORS(app)

# ─────────────────────────────────────────────
# MongoDB Setup
# ─────────────────────────────────────────────
MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/')
mongo_client = None
db = None
users_col = None
sessions_col = None

try:
    mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
    mongo_client.server_info()  # Test connection
    db = mongo_client['signease']
    users_col = db['users']
    sessions_col = db['sessions']
    users_col.create_index('email', unique=True)
    print("✓ MongoDB connected to 'signease' database")
except Exception as e:
    print(f"⚠ MongoDB connection failed: {e}")
    print("  Make sure MongoDB is running on localhost:27017")


# ─────────────────────────────────────────────
# Auth Helper
# ─────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('user_id'):
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated

# ─────────────────────────────────────────────
# NVIDIA NIM AI Client
# ─────────────────────────────────────────────
NVIDIA_NIM_API_KEY = os.environ.get('NVIDIA_NIM_API_KEY', 'your_nvidia_nim_api_key_here')
NVIDIA_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_NIM_MODEL = "nvidia/llama-3.1-nemotron-nano-8b-v1"

nim_client = None
try:
    if NVIDIA_NIM_API_KEY and NVIDIA_NIM_API_KEY != 'your_nvidia_nim_api_key_here':
        nim_client = OpenAI(
            base_url=NVIDIA_NIM_BASE_URL,
            api_key=NVIDIA_NIM_API_KEY
        )
        print("✓ NVIDIA NIM client initialized")
    else:
        print("⚠ NVIDIA_NIM_API_KEY not set. Add it to your .env file.")
except Exception as e:
    print(f"⚠ NVIDIA NIM init failed: {e}")

# ─────────────────────────────────────────────
# TTS Engine (shared across modes)
# ─────────────────────────────────────────────
try:
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', 150)
    print("✓ TTS engine initialized")
except Exception as e:
    print(f"⚠ TTS init failed: {e}")
    tts_engine = None

tts_lock = threading.Lock()

# ─────────────────────────────────────────────
# ASL Model & Config
# ─────────────────────────────────────────────
ASL_MODEL_PATH = os.path.join('models', 'asl_model.p')
LANDMARKER_PATH = os.path.join('models', 'hand_landmarker.task')

asl_model = None
try:
    model_dict = pickle.load(open(ASL_MODEL_PATH, 'rb'))
    for key in ['model6', 'model', 'classifier']:
        if key in model_dict:
            asl_model = model_dict[key]
            break
    if asl_model is None:
        asl_model = list(model_dict.values())[0]
    print(f"✓ ASL model loaded from {ASL_MODEL_PATH}")
except Exception as e:
    print(f"⚠ ASL model load failed: {e}")

ASL_LABELS = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
    8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P',
    16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X',
    24: 'Y', 25: ' '
}

# ─────────────────────────────────────────────
# ISL Model & Config
# ─────────────────────────────────────────────
ISL_MODEL_PATH = os.path.join('models', 'isl_model.h5')
ISL_LABELS_PATH = os.path.join('models', 'isl_label_classes.json')

isl_model = None
isl_alphabet = []

try:
    if os.path.exists(ISL_MODEL_PATH):
        isl_model = keras.models.load_model(ISL_MODEL_PATH)
        print(f"✓ ISL model loaded from {ISL_MODEL_PATH}")
    else:
        print(f"⚠ ISL model not found: {ISL_MODEL_PATH}")

    if os.path.exists(ISL_LABELS_PATH):
        with open(ISL_LABELS_PATH, 'r') as f:
            isl_alphabet = json.load(f)
        print(f"✓ ISL labels loaded: {len(isl_alphabet)} classes")
    else:
        isl_alphabet = [str(i) for i in range(1, 10)] + list(string.ascii_uppercase)
        print("⚠ ISL labels not found, using fallback")
except Exception as e:
    print(f"⚠ ISL init error: {e}")


# ─────────────────────────────────────────────
# Global State
# ─────────────────────────────────────────────
current_mode = "ASL"  # "ASL" or "ISL"
current_prediction = ""
accumulated_sentence = ""
last_prediction_time = 0
stop_signal = False

# ISL specific
COOLDOWN_ISL = 1.0
STABILITY_THRESHOLD = 5
prediction_buffer = []

# ASL specific
COOLDOWN_ASL = 3.0


# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────
def normalize_asl_landmarks(hand_landmarks, x_min, y_min):
    """Normalize ASL hand landmarks relative to bounding box origin."""
    return [coord for lm in hand_landmarks for coord in (lm.x - x_min, lm.y - y_min)]


def calc_landmark_list_isl(hand_landmarks, img_w, img_h):
    """Convert Tasks API landmarks to pixel-space list for ISL."""
    landmark_point = []
    for lm in hand_landmarks:
        lx = min(int(lm.x * img_w), img_w - 1)
        ly = min(int(lm.y * img_h), img_h - 1)
        landmark_point.append([lx, ly])
    return landmark_point


def pre_process_landmark_isl(landmark_list):
    """Pre-process ISL landmarks: relative coords + normalization."""
    temp = copy.deepcopy(landmark_list)
    base_x, base_y = 0, 0
    for index, point in enumerate(temp):
        if index == 0:
            base_x, base_y = point[0], point[1]
        temp[index][0] -= base_x
        temp[index][1] -= base_y
    temp = list(itertools.chain.from_iterable(temp))
    max_val = max(list(map(abs, temp))) if temp else 1
    return [n / max_val if max_val != 0 else 0 for n in temp]


def draw_hand_landmarks(frame, hand_landmarks, W, H, color=(0, 255, 0)):
    """Draw hand landmarks and connections on frame."""
    # MediaPipe hand connections
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),       # Index
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
        (0, 13), (13, 14), (14, 15), (15, 16),# Ring
        (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
        (5, 9), (9, 13), (13, 17)              # Palm
    ]
    points = []
    for lm in hand_landmarks:
        x = int(lm.x * W)
        y = int(lm.y * H)
        points.append((x, y))
        cv2.circle(frame, (x, y), 4, color, -1)

    for start, end in HAND_CONNECTIONS:
        if start < len(points) and end < len(points):
            cv2.line(frame, points[start], points[end], color, 2)


# ─────────────────────────────────────────────
# Create HandLandmarker detector
# ─────────────────────────────────────────────
def create_detector(num_hands=1):
    """Create a MediaPipe HandLandmarker instance."""
    try:
        options = hand_landmarker.HandLandmarkerOptions(
            base_options=base_options.BaseOptions(
                model_asset_path=LANDMARKER_PATH
            ),
            num_hands=num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        detector = hand_landmarker.HandLandmarker.create_from_options(options)
        print(f"✓ HandLandmarker created (num_hands={num_hands})")
        return detector
    except Exception as e:
        print(f"⚠ HandLandmarker init error: {e}")
        return None


# ─────────────────────────────────────────────
# ASL Frame Generator
# ─────────────────────────────────────────────
def gen_frames_asl():
    global current_prediction, accumulated_sentence, stop_signal, last_prediction_time

    detector = create_detector(num_hands=1)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        for i in range(1, 5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                break
        else:
            print("ERROR: No camera available!")
            return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print(f"✓ ASL camera ready: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    frame_count = 0

    while not stop_signal:
        ret, frame = cap.read()
        if not ret:
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        current_time = time.time()
        predicted_character = ""

        if detector is not None and asl_model is not None:
            try:
                mp_image = MpImage(image_format=ImageFormat.SRGB, data=frame_rgb)
                detection_result = detector.detect(mp_image)

                if detection_result.hand_landmarks and len(detection_result.hand_landmarks) > 0:
                    hand_lms = detection_result.hand_landmarks[0]

                    # Draw landmarks
                    draw_hand_landmarks(frame, hand_lms, W, H, (0, 255, 0))

                    # Get bounding box
                    x_coords = [lm.x for lm in hand_lms]
                    y_coords = [lm.y for lm in hand_lms]
                    data_aux = normalize_asl_landmarks(hand_lms, min(x_coords), min(y_coords))

                    if len(data_aux) == 42:
                        prediction = asl_model.predict([np.asarray(data_aux)])
                        predicted_character = ASL_LABELS.get(int(prediction[0]), "")
                        current_prediction = predicted_character

                        # Draw bounding box and label
                        x1 = int(min(x_coords) * W) - 10
                        y1 = int(min(y_coords) * H) - 10
                        x2 = int(max(x_coords) * W) + 10
                        y2 = int(max(y_coords) * H) + 10
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        cv2.putText(frame, predicted_character, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

                        # Add to sentence with cooldown
                        if current_time - last_prediction_time > COOLDOWN_ASL:
                            if predicted_character.strip():
                                accumulated_sentence += predicted_character
                                last_prediction_time = current_time
                else:
                    current_prediction = ""

            except Exception as e:
                if frame_count == 0:
                    print(f"ASL detection error: {e}")
        else:
            if frame_count == 0:
                cv2.putText(frame, "ASL Model Unavailable", (50, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        frame_count += 1

    cap.release()
    if detector:
        try:
            detector.close()
        except:
            pass


# ─────────────────────────────────────────────
# ISL Frame Generator
# ─────────────────────────────────────────────
def gen_frames_isl():
    global current_prediction, accumulated_sentence, stop_signal, last_prediction_time, prediction_buffer

    detector = create_detector(num_hands=2)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        for i in range(1, 5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                break
        else:
            print("ERROR: No camera available!")
            return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print(f"✓ ISL camera ready: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    while not stop_signal:
        success, image = cap.read()
        if not success:
            break

        image = cv2.flip(image, 1)
        H, W, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = ""

        if detector is not None and isl_model is not None:
            try:
                mp_image = MpImage(image_format=ImageFormat.SRGB, data=image_rgb)
                detection_result = detector.detect(mp_image)

                if detection_result.hand_landmarks and len(detection_result.hand_landmarks) > 0:
                    all_landmarks = [0.0] * 84

                    for hand_idx, hand_lms in enumerate(detection_result.hand_landmarks):
                        if hand_idx >= 2:
                            break

                        # Draw landmarks
                        colors = [(0, 255, 0), (255, 165, 0)]
                        draw_hand_landmarks(image, hand_lms, W, H, colors[hand_idx % 2])

                        # Process keypoints
                        landmark_list = calc_landmark_list_isl(hand_lms, W, H)
                        pre_processed = pre_process_landmark_isl(landmark_list)
                        start_idx = hand_idx * 42
                        all_landmarks[start_idx:start_idx + 42] = pre_processed

                    try:
                        prediction = isl_model.predict(np.array([all_landmarks]), verbose=0)
                        idx = np.argmax(prediction[0])
                        label = isl_alphabet[idx]
                        current_prediction = label

                        # Stability buffer logic
                        prediction_buffer.append(label)
                        if len(prediction_buffer) > STABILITY_THRESHOLD:
                            prediction_buffer.pop(0)

                        # Check stability + cooldown
                        if (len(prediction_buffer) == STABILITY_THRESHOLD and
                                all(x == label for x in prediction_buffer)):
                            now = time.time()
                            if now - last_prediction_time > COOLDOWN_ISL:
                                if label == '1':
                                    accumulated_sentence += " "
                                elif label == '2':
                                    words = accumulated_sentence.rstrip().split(' ')
                                    if words:
                                        accumulated_sentence = " ".join(words[:-1])
                                        if accumulated_sentence:
                                            accumulated_sentence += " "
                                    else:
                                        accumulated_sentence = ""
                                else:
                                    accumulated_sentence += label
                                last_prediction_time = now
                                prediction_buffer.clear()
                    except Exception as e:
                        print(f"ISL prediction error: {e}")
                else:
                    current_prediction = ""
                    prediction_buffer.clear()

            except Exception as e:
                print(f"ISL detection error: {e}")
        else:
            cv2.putText(image, "ISL Model Unavailable", (50, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', image)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()
    if detector:
        try:
            detector.close()
        except:
            pass


# ═════════════════════════════════════════════
# Flask Routes
# ═════════════════════════════════════════════
# ─────────────────────────────────────────────
# Auth Routes
# ─────────────────────────────────────────────
@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if session.get('user_id'):
        return redirect(url_for('index'))
    error = None
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        if not email or not password:
            error = 'Please fill in all fields.'
        elif users_col is None:
            error = 'Database not available. Is MongoDB running?'
        else:
            user = users_col.find_one({'email': email})
            if user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
                session['user_id'] = str(user['_id'])
                session['user_name'] = user['name']
                session['user_email'] = user['email']
                return redirect(url_for('index'))
            else:
                error = 'Invalid email or password.'
    return render_template('login.html', error=error)


@app.route('/register', methods=['GET', 'POST'])
def register_page():
    if session.get('user_id'):
        return redirect(url_for('index'))
    error = None
    success = None
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm = request.form.get('confirm_password', '')
        if not name or not email or not password or not confirm:
            error = 'Please fill in all fields.'
        elif password != confirm:
            error = 'Passwords do not match.'
        elif len(password) < 6:
            error = 'Password must be at least 6 characters.'
        elif users_col is None:
            error = 'Database not available. Is MongoDB running?'
        else:
            try:
                hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
                users_col.insert_one({
                    'name': name,
                    'email': email,
                    'password': hashed,
                    'created_at': datetime.utcnow()
                })
                success = 'Account created! You can now log in.'
            except Exception as e:
                if 'duplicate' in str(e).lower() or '11000' in str(e):
                    error = 'An account with this email already exists.'
                else:
                    error = f'Registration failed: {str(e)}'
    return render_template('register.html', error=error, success=success)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login_page'))


# ─────────────────────────────────────────────
# Main App Route (auth-protected)
# ─────────────────────────────────────────────
@app.route('/')
@login_required
def index():
    return render_template('index.html',
                           user_name=session.get('user_name', 'User'),
                           user_email=session.get('user_email', ''))


@app.route('/video_feed')
def video_feed():
    global stop_signal, current_prediction, accumulated_sentence, prediction_buffer
    stop_signal = False
    current_prediction = ""
    prediction_buffer = []

    if current_mode == "ISL":
        return Response(gen_frames_isl(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response(gen_frames_asl(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_data')
def get_data():
    return jsonify({
        "prediction": current_prediction,
        "sentence": accumulated_sentence,
        "mode": current_mode
    })


@app.route('/switch_mode', methods=['POST'])
def switch_mode():
    global current_mode, stop_signal, current_prediction, accumulated_sentence, prediction_buffer
    data = request.get_json()
    new_mode = data.get('mode', 'ASL').upper()
    if new_mode in ['ASL', 'ISL']:
        stop_signal = True
        time.sleep(0.5)
        current_mode = new_mode
        current_prediction = ""
        accumulated_sentence = ""
        prediction_buffer = []
        print(f"✓ Switched to {current_mode} mode")
        return jsonify({"status": "success", "mode": current_mode})
    return jsonify({"status": "error", "message": "Invalid mode"}), 400


@app.route('/clear_sentence', methods=['POST'])
def clear_sentence():
    global accumulated_sentence, current_prediction, prediction_buffer
    accumulated_sentence = ""
    current_prediction = ""
    prediction_buffer = []
    return jsonify({"status": "success"})


@app.route('/speak', methods=['POST'])
def speak():
    if not accumulated_sentence.strip():
        return jsonify({"status": "error", "message": "Nothing to speak"})
    text_to_speak = accumulated_sentence  # Capture current value

    def _speak():
        with tts_lock:
            try:
                # Spawn a completely separate Python process each time.
                # pyttsx3.init() on Windows returns a CACHED engine singleton
                # from its internal _activeEngines dict, so even calling init()
                # again gives back the broken stuck engine from before.
                # A fresh subprocess bypasses this entirely and always works.
                script = (
                    "import pyttsx3; "
                    "e = pyttsx3.init(); "
                    "e.setProperty('rate', 150); "
                    f"e.say({repr(text_to_speak)}); "
                    "e.runAndWait()"
                )
                # CREATE_NO_WINDOW = 0x08000000 prevents a console flash on Windows
                flags = 0x08000000 if sys.platform == 'win32' else 0
                subprocess.run(
                    [sys.executable, "-c", script],
                    timeout=30,
                    creationflags=flags
                )
            except Exception as e:
                print(f"TTS error: {e}")

    threading.Thread(target=_speak, daemon=True).start()
    return jsonify({"status": "success"})


@app.route('/stop', methods=['POST'])
def stop():
    global stop_signal
    stop_signal = True
    return jsonify({"status": "success"})


@app.route('/delete_last', methods=['POST'])
def delete_last():
    global accumulated_sentence
    if accumulated_sentence:
        accumulated_sentence = accumulated_sentence[:-1]
    return jsonify({"status": "success", "sentence": accumulated_sentence})


@app.route('/add_space', methods=['POST'])
def add_space():
    global accumulated_sentence
    accumulated_sentence += " "
    return jsonify({"status": "success", "sentence": accumulated_sentence})


@app.route('/correct', methods=['POST'])
def correct_sentence():
    """Use NVIDIA NIM (Llama) to correct the accumulated sentence."""
    global accumulated_sentence
    raw_text = accumulated_sentence.strip()
    if not raw_text:
        return jsonify({"status": "error", "message": "No text to correct"})
    if not nim_client:
        return jsonify({"status": "error", "message": "NVIDIA NIM API key not configured"})

    try:
        prompt = (
            f"The following text was typed using sign language gestures, "
            f"so it may have spelling mistakes or jumbled letters. "
            f"Correct it into the most likely intended word or sentence. "
            f"ONLY return the corrected text, nothing else — no quotes, no explanation.\n\n"
            f"{raw_text}"
        )
        response = nim_client.chat.completions.create(
            model=NVIDIA_NIM_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful text correction assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=256
        )
        corrected = response.choices[0].message.content.strip()
        accumulated_sentence = corrected
        return jsonify({"status": "success", "corrected": corrected, "original": raw_text})
    except Exception as e:
        print(f"NVIDIA NIM error: {e}")
        return jsonify({"status": "error", "message": str(e)})


@app.route('/model_status')
def model_status():
    return jsonify({
        "asl_model": asl_model is not None,
        "isl_model": isl_model is not None,
        "tts_engine": tts_engine is not None,
        "nvidia_nim": nim_client is not None,
        "current_mode": current_mode
    })


# ─────────────────────────────────────────────
# User Session Save / History Routes
# ─────────────────────────────────────────────
@app.route('/save_session', methods=['POST'])
@login_required
def save_session_route():
    global accumulated_sentence
    text = accumulated_sentence.strip()
    if not text:
        return jsonify({'status': 'error', 'message': 'Nothing to save — sentence is empty.'})
    if sessions_col is None:
        return jsonify({'status': 'error', 'message': 'Database not available.'})
    try:
        sessions_col.insert_one({
            'user_id': session['user_id'],
            'user_email': session.get('user_email', ''),
            'sentence': text,
            'mode': current_mode,
            'saved_at': datetime.utcnow()
        })
        return jsonify({'status': 'success', 'message': 'Session saved!'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/user_history', methods=['GET'])
@login_required
def user_history():
    if sessions_col is None:
        return jsonify({'status': 'error', 'items': []})
    try:
        uid = session['user_id']
        records = list(sessions_col.find(
            {'user_id': uid},
            {'_id': 0, 'sentence': 1, 'mode': 1, 'saved_at': 1}
        ).sort('saved_at', -1).limit(20))
        # Convert datetime to string for JSON serialisation
        for r in records:
            if 'saved_at' in r:
                r['saved_at'] = r['saved_at'].strftime('%d %b %Y, %I:%M %p')
        return jsonify({'status': 'success', 'items': records})
    except Exception as e:
        return jsonify({'status': 'error', 'items': [], 'message': str(e)})


# ─────────────────────────────────────────────
app.jinja_env.auto_reload = True
app.config['TEMPLATES_AUTO_RELOAD'] = True

if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("  Unified Sign Language Detection System")
    print("  ASL Model:", "✓ Loaded" if asl_model else "✗ Not loaded")
    print("  ISL Model:", "✓ Loaded" if isl_model else "✗ Not loaded")
    print("  TTS Engine:", "✓ Ready" if tts_engine else "✗ Not available")
    print("=" * 50)
    print("  Open http://localhost:5050 in your browser")
    print("=" * 50 + "\n")
    app.run(host='0.0.0.0', port=5050, debug=False)
