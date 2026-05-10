"""
Train ISL Model - Neural Network Training Pipeline
Trains a Sequential NN on extracted hand landmarks (84 features).
Outputs: models/isl_model.h5 and models/isl_label_classes.json

Usage: python train_model.py
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import json
import traceback
import sys

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)


def load_data(csv_path):
    """Load keypoint data from CSV. Format: Label, x1, y1, x2, y2, ... (84 features)."""
    print(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path, header=None)
        X = df.iloc[:, 1:].values
        y = df.iloc[:, 0].values

        if np.any(np.isnan(X)):
            print("Warning: NaN values found. Cleaning...")
            X = np.nan_to_num(X)

        print(f"Data loaded. Shape: X={X.shape}, y={y.shape}")
        return X, y
    except Exception as e:
        print(f"Error loading data: {e}")
        traceback.print_exc()
        return None, None


def create_model(input_shape, num_classes):
    """Create a robust NN model with Dropout and BatchNorm."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=input_shape),

        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(32, activation='relu'),

        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model


def train():
    """Main training function."""
    csv_path = 'keypoint.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found! Run generate_keypoints.py first.")
        return

    X, y = load_data(csv_path)
    if X is None:
        return

    # Force labels to string
    y = y.astype(str)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    print(f"Classes detected ({num_classes}): {label_encoder.classes_}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    model = create_model(input_shape=(84,), num_classes=num_classes)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # Callbacks
    os.makedirs('models', exist_ok=True)
    model_save_path = os.path.join('models', 'isl_model.h5')
    label_save_path = os.path.join('models', 'isl_label_classes.json')

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            model_save_path, monitor='val_loss',
            save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=5, min_lr=0.00001, verbose=1
        )
    ]

    model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )

    # Evaluate
    print("\n--- Final Evaluation ---")
    val_loss, val_acc = model.evaluate(X_val, y_val)
    print(f"Validation Accuracy: {val_acc:.4f}")

    # Save label classes
    classes_list = label_encoder.classes_.tolist()
    with open(label_save_path, 'w') as f:
        json.dump(classes_list, f)
    print(f"Label classes saved to '{label_save_path}': {classes_list}")
    print(f"Model saved to '{model_save_path}'")


if __name__ == '__main__':
    try:
        train()
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
