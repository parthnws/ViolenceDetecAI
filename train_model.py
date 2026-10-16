"""Train the violence classifier (MobileNetV2 + LSTM) on clips.

Dataset layout (e.g. RWF-2000 or Real Life Violence Situations from Kaggle):
    data/Violence/*.avi|mp4
    data/NonViolence/*.avi|mp4

Run:  python train_model.py --data data --epochs 15
"""
import argparse
import glob
import os

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from detector import IMG_SIZE, SEQ_LEN


def load_clip(path):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = set(np.linspace(0, max(n - 1, 0), SEQ_LEN).astype(int))
    frames, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i in idx:
            f = cv2.cvtColor(cv2.resize(f, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB)
            frames.append(f.astype("float32") / 255.0)
        i += 1
    cap.release()
    while 0 < len(frames) < SEQ_LEN:
        frames.append(frames[-1])
    return np.array(frames[:SEQ_LEN]) if frames else None


def build_model():
    base = tf.keras.applications.MobileNetV2(
        include_top=False, weights="imagenet", pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False
    m = tf.keras.Sequential([
        tf.keras.layers.Input((SEQ_LEN, IMG_SIZE, IMG_SIZE, 3)),
        tf.keras.layers.TimeDistributed(base),
        tf.keras.layers.LSTM(128),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    m.compile("adam", "binary_crossentropy", metrics=["accuracy"])
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--epochs", type=int, default=15)
    a = ap.parse_args()

    X, y = [], []
    for label, folder in enumerate(["NonViolence", "Violence"]):
        for p in glob.glob(os.path.join(a.data, folder, "*")):
            c = load_clip(p)
            if c is not None:
                X.append(c); y.append(label)
    X, y = np.array(X), np.array(y)
    print("Clips:", X.shape, "violent:", int(y.sum()))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = build_model()
    os.makedirs("models", exist_ok=True)
    model.fit(Xtr, ytr, validation_data=(Xte, yte), epochs=a.epochs, batch_size=8,
              callbacks=[tf.keras.callbacks.ModelCheckpoint(
                  "models/violence_model.keras", save_best_only=True, monitor="val_accuracy")])
