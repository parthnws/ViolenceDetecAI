"""Core detection logic: violence scoring, weapon detection, severity levels."""
from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

SEQ_LEN = 16          # frames per clip fed to the violence model
IMG_SIZE = 128        # must match train_model.py

LEVELS = {
    0: ("SAFE", "#2e7d32"),
    1: ("LEVEL 1 - LOW", "#f9a825"),
    2: ("LEVEL 2 - MEDIUM", "#ef6c00"),
    3: ("LEVEL 3 - HIGH", "#c62828"),
}

# Weapon class names (as in your YOLO weapon model) -> danger weight
WEAPON_WEIGHT = {
    "gun": 3, "pistol": 3, "rifle": 3, "firearm": 3, "handgun": 3,
    "knife": 2, "machete": 2, "sword": 2, "axe": 2,
    "bat": 1, "stick": 1, "rod": 1, "hammer": 1,
}


@dataclass
class FrameResult:
    frame: np.ndarray
    violence_score: float
    level: int
    weapons: list = field(default_factory=list)   # [(name, conf, (x1,y1,x2,y2))]
    mode: str = "demo"


class ViolenceDetector:
    def __init__(self, violence_model_path="models/violence_model.keras",
                 weapon_model_path="models/weapon.pt",
                 t1=0.50, t2=0.70, t3=0.88):
        self.t1, self.t2, self.t3 = t1, t2, t3
        self.buffer: deque = deque(maxlen=SEQ_LEN)
        self.prev_gray = None
        self.smooth: deque = deque(maxlen=5)
        self.model = None
        self.weapon_model = None
        self.mode = "demo"

        if os.path.exists(violence_model_path):
            import tensorflow as tf
            self.model = tf.keras.models.load_model(violence_model_path)
            self.mode = "model"
        if os.path.exists(weapon_model_path):
            try:
                from ultralytics import YOLO
                self.weapon_model = YOLO(weapon_model_path)
            except Exception:
                self.weapon_model = None

    # ---- violence score -------------------------------------------------
    def _motion_score(self, frame) -> float:
        """Demo fallback: erratic, high-energy motion heuristic (NOT a real classifier)."""
        gray = cv2.cvtColor(cv2.resize(frame, (160, 120)), cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if self.prev_gray is None:
            self.prev_gray = gray
            return 0.0
        flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None,
                                            0.5, 2, 15, 2, 5, 1.1, 0)
        self.prev_gray = gray
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        energy = float(np.mean(mag))
        active = float(np.mean(mag > 1.5))
        return float(np.clip(0.5 * energy / 3.0 + 0.5 * active * 2, 0, 1))

    def _model_score(self, frame) -> float:
        img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
        self.buffer.append(img)
        if len(self.buffer) < SEQ_LEN:
            return 0.0
        clip = np.expand_dims(np.array(self.buffer), 0)
        return float(self.model.predict(clip, verbose=0)[0][0])

    # ---- weapons --------------------------------------------------------
    def _detect_weapons(self, frame, conf=0.45):
        out = []
        if self.weapon_model is None:
            return out
        res = self.weapon_model.predict(frame, conf=conf, verbose=False)[0]
        for b in res.boxes:
            name = res.names[int(b.cls[0])].lower()
            out.append((name, float(b.conf[0]),
                        tuple(int(v) for v in b.xyxy[0].tolist())))
        return out

    # ---- severity -------------------------------------------------------
    def _level(self, score, weapons) -> int:
        if score >= self.t3:
            lvl = 3
        elif score >= self.t2:
            lvl = 2
        elif score >= self.t1:
            lvl = 1
        else:
            lvl = 0
        if weapons:
            w = max(WEAPON_WEIGHT.get(n, 2) for n, _, _ in weapons)
            if w >= 3:
                lvl = 3                      # firearm = always high
            elif lvl > 0 or w >= 2:
                lvl = max(lvl, 2)            # bladed weapon escalates
            else:
                lvl = max(lvl, 1)
        return lvl

    # ---- public ---------------------------------------------------------
    def process(self, frame) -> FrameResult:
        raw = self._model_score(frame) if self.model else self._motion_score(frame)
        self.smooth.append(raw)
        score = float(np.mean(self.smooth))
        weapons = self._detect_weapons(frame)
        level = self._level(score, weapons)
        return FrameResult(self.annotate(frame, score, level, weapons),
                           score, level, weapons, self.mode)

    @staticmethod
    def annotate(frame, score, level, weapons):
        out = frame.copy()
        name, hexcol = LEVELS[level]
        bgr = tuple(int(hexcol[i:i + 2], 16) for i in (5, 3, 1))
        for wname, wconf, (x1, y1, x2, y2) in weapons:
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(out, f"{wname} {wconf:.0%}", (x1, max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.rectangle(out, (0, 0), (out.shape[1], 34), bgr, -1)
        cv2.putText(out, f"{name}  |  score {score:.2f}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if level:
            cv2.rectangle(out, (0, 0), (out.shape[1] - 1, out.shape[0] - 1), bgr, 4)
        return out
