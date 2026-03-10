"""
hand_detector.py
─────────────────────────────────────────────
Detects hand gestures using MediaPipe HandLandmarker
21 landmark points — Tasks API compatible
─────────────────────────────────────────────

HAND LANDMARKS:
    0=WRIST  4=THUMB_TIP  8=INDEX_TIP
    12=MIDDLE_TIP  16=RING_TIP  20=PINKY_TIP
─────────────────────────────────────────────
"""

import cv2
import numpy as np
import os
import pickle
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode


# ── Auto-download model ──
_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "models", "hand_landmarker.task"
)
os.makedirs(os.path.dirname(_MODEL_PATH), exist_ok=True)

if not os.path.exists(_MODEL_PATH):
    print("📥 Downloading hand landmark model (first time only)...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        _MODEL_PATH
    )
    print("✅ Hand model downloaded")
else:
    print("✅ Hand model found")


class HandGestureDetector:
    def __init__(self):
        # ── Landmark names ──
        self.LANDMARK_NAMES = {
            0:"WRIST",
            1:"THUMB_CMC",  2:"THUMB_MCP",  3:"THUMB_IP",   4:"THUMB_TIP",
            5:"INDEX_MCP",  6:"INDEX_PIP",  7:"INDEX_DIP",  8:"INDEX_TIP",
            9:"MIDDLE_MCP", 10:"MIDDLE_PIP",11:"MIDDLE_DIP",12:"MIDDLE_TIP",
            13:"RING_MCP",  14:"RING_PIP",  15:"RING_DIP",  16:"RING_TIP",
            17:"PINKY_MCP", 18:"PINKY_PIP", 19:"PINKY_DIP", 20:"PINKY_TIP",
        }
        self.FINGER_TIPS   = [4, 8, 12, 16, 20]
        self.FINGER_BASES  = [2, 5,  9, 13, 17]
        self.FINGER_MIDDLE = [3, 7, 11, 15, 19]

        # Hand connections for drawing
        self.CONNECTIONS = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),
            (0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20),
            (5,9),(9,13),(13,17)
        ]

        # ── Load ML model if available ──
        self.model = None
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "models", "gesture_model.pkl"
        )
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            print("✅ Gesture ML model loaded")
        else:
            print("⚠️  No gesture model — using rule-based detection")

        # ── Setup HandLandmarker ──
        options = HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=VisionTaskRunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hand_landmarker = HandLandmarker.create_from_options(options)
        print("✅ HandGestureDetector ready")

    # ─────────────────────────────────────
    #  LANDMARK WRAPPER
    # ─────────────────────────────────────
    class _LM:
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z

    # ─────────────────────────────────────
    #  MAIN DETECTION
    # ─────────────────────────────────────
    def detect(self, frame):
        """
        Returns: (gesture_label, confidence, annotated_frame)
        """
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = self.hand_landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return None, 0.0, frame

        h, w, _ = frame.shape

        # ── Draw all hands ──
        for hand_lm in result.hand_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lm]
            for pt in pts:
                cv2.circle(frame, pt, 5, (0, 200, 255), -1)
            for a, b in self.CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], (0, 255, 200), 2)

        # ── Use first hand for detection ──
        landmarks = [self._LM(lm.x, lm.y, lm.z)
                     for lm in result.hand_landmarks[0]]
        features  = self._extract_features(landmarks)

        if self.model:
            gesture, confidence = self._ml_predict(features)
        else:
            gesture, confidence = self._rule_based_detect(landmarks)

        return gesture, confidence, frame

    # ─────────────────────────────────────
    #  FEATURE EXTRACTION
    # ─────────────────────────────────────
    def _extract_features(self, landmarks):
        wrist    = landmarks[0]
        features = []
        for lm in landmarks:
            features.extend([
                lm.x - wrist.x,
                lm.y - wrist.y,
                lm.z - wrist.z
            ])
        return np.array(features, dtype=np.float32)

    # ─────────────────────────────────────
    #  FINGER STATE HELPERS
    # ─────────────────────────────────────
    def _get_finger_states(self, landmarks):
        states = []
        # Thumb (x-axis comparison)
        states.append(landmarks[4].x < landmarks[3].x)
        # Other 4 fingers (y-axis: tip above pip = up)
        for tip, base in zip([8, 12, 16, 20], [6, 10, 14, 18]):
            states.append(landmarks[tip].y < landmarks[base].y)
        return states  # [thumb, index, middle, ring, pinky]

    def _fingers_up_count(self, landmarks):
        return sum(self._get_finger_states(landmarks))

    # ─────────────────────────────────────
    #  RULE-BASED GESTURE DETECTION
    # ─────────────────────────────────────
    def _rule_based_detect(self, landmarks):
        states = self._get_finger_states(landmarks)
        thumb, index, middle, ring, pinky = states
        count = sum(states)

        # 👍 Thumbs up = YES
        if thumb and not index and not middle and not ring and not pinky:
            return "yes", 0.85

        # ✋ All 5 open = HELP
        if count == 5:
            return "help", 0.85

        # ☝️ Index only = WATER
        if not thumb and index and not middle and not ring and not pinky:
            return "water", 0.75

        # ✌️ Index + Middle = FOOD or PAIN
        if not thumb and index and middle and not ring and not pinky:
            idx_tip = np.array([landmarks[8].x, landmarks[8].y])
            mid_tip = np.array([landmarks[12].x, landmarks[12].y])
            if np.linalg.norm(idx_tip - mid_tip) < 0.05:
                return "pain", 0.70
            return "food", 0.78

        # 🤙 Thumb + Pinky = HELP
        if thumb and not index and not middle and not ring and pinky:
            return "help", 0.80

        # 👌 Pinch = YES
        thumb_tip  = np.array([landmarks[4].x, landmarks[4].y])
        index_tip  = np.array([landmarks[8].x, landmarks[8].y])
        pinch_dist = np.linalg.norm(thumb_tip - index_tip)
        if pinch_dist < 0.05 and middle and ring and pinky:
            return "yes", 0.85

        # ✊ Fist = PAIN
        if count == 0:
            return "pain", 0.65

        # 👎 Closed low = NO
        wrist_y = landmarks[0].y
        if all(wrist_y < landmarks[i].y for i in [8,12,16,20]) and count == 0:
            return "no", 0.80

        return None, 0.0

    # ─────────────────────────────────────
    #  ML MODEL PREDICTION
    # ─────────────────────────────────────
    def _ml_predict(self, features):
        try:
            pred       = self.model.predict([features])[0]
            proba      = self.model.predict_proba([features])[0]
            confidence = float(np.max(proba))
            return (pred, confidence) if confidence >= 0.5 else (None, confidence)
        except Exception:
            return None, 0.0

    # ─────────────────────────────────────
    #  GET RAW LANDMARKS
    # ─────────────────────────────────────
    def get_all_landmarks(self, frame):
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = self.hand_landmarker.detect(mp_image)
        if not result.hand_landmarks:
            return None
        return [(lm.x, lm.y, lm.z) for lm in result.hand_landmarks[0]]