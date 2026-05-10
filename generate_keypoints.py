"""
Generate Keypoints - ISL Training Data Extraction
Extracts MediaPipe hand landmarks from dataset images and saves to CSV.
Supports 2-hand detection (84 features: 42 per hand).

Usage: python generate_keypoints.py
"""

import cv2
import mediapipe as mp
import csv
import copy
import itertools
import os
import glob
import random
import string

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands


def calc_landmark_list(image, landmarks):
    """Convert normalized landmarks to pixel coordinates."""
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point


def pre_process_landmark(landmark_list):
    """Convert to relative coordinates and normalize."""
    temp = copy.deepcopy(landmark_list)

    # Convert to relative (wrist as origin)
    base_x, base_y = 0, 0
    for index, point in enumerate(temp):
        if index == 0:
            base_x, base_y = point[0], point[1]
        temp[index][0] -= base_x
        temp[index][1] -= base_y

    # Flatten
    temp = list(itertools.chain.from_iterable(temp))

    # Normalize
    max_value = max(list(map(abs, temp))) if temp else 1

    def normalize_(n):
        return n / max_value if max_value != 0 else 0

    return list(map(normalize_, temp))


def process_dataset():
    """Process dataset images and extract hand landmark features to CSV."""
    csv_path = 'keypoint.csv'

    if os.path.exists(csv_path):
        os.remove(csv_path)
        print(f"Removed old {csv_path}")

    # Configure your dataset directories here
    base_dir = 'dataset from kaggle'

    data_map = {}

    # Add numbers 1-9
    for i in range(1, 10):
        label = str(i)
        path = os.path.join(base_dir, label)
        if os.path.exists(path):
            data_map[label] = path
        else:
            print(f"Warning: Path not found for {label}: {path}")

    # Add Alphabets A-Z
    indian_dir = os.path.join(base_dir, 'Indian')
    for char in string.ascii_uppercase:
        path = os.path.join(indian_dir, char)
        if os.path.exists(path):
            data_map[char] = path
        else:
            print(f"Warning: Path not found for {char}: {path}")

    print(f"Found {len(data_map)} classes to process: {sorted(data_map.keys())}")

    # Process with MediaPipe Hands
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5
    ) as hands:

        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)

            for label, folder_path in data_map.items():
                print(f"Processing class: {label}")

                image_files = (
                    glob.glob(os.path.join(folder_path, '*.jpg')) +
                    glob.glob(os.path.join(folder_path, '*.jpeg')) +
                    glob.glob(os.path.join(folder_path, '*.png'))
                )

                random.shuffle(image_files)
                count = 0
                total = len(image_files)

                for idx, image_path in enumerate(image_files):
                    if idx % 200 == 0:
                        print(f"    Processing image {idx}/{total}")
                    try:
                        image = cv2.imread(image_path)
                        if image is None:
                            continue

                        image = cv2.flip(image, 1)
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        results = hands.process(image_rgb)

                        if results.multi_hand_landmarks:
                            all_landmarks = [0.0] * 84

                            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                                if hand_idx >= 2:
                                    break
                                landmark_list = calc_landmark_list(image, hand_landmarks)
                                pre_processed = pre_process_landmark(landmark_list)
                                start_idx = hand_idx * 42
                                all_landmarks[start_idx:start_idx + 42] = pre_processed

                            writer.writerow([label, *all_landmarks])
                            count += 1
                    except Exception as e:
                        print(f"Error processing {image_path}: {e}")

                print(f"  - Extracted {count} samples for {label}")

    print(f"Finished! Data saved to {csv_path}")


if __name__ == "__main__":
    process_dataset()
