"""
face_detector.py
─────────────────────────────────────────────
80 Rule-Based Face Expression Detectors
Using MediaPipe FaceLandmarker (Tasks API)
Compatible with mediapipe 0.10.30+
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
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode


# ── Auto-download model ──
_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "models", "face_landmarker.task"
)
os.makedirs(os.path.dirname(_MODEL_PATH), exist_ok=True)

if not os.path.exists(_MODEL_PATH):
    print("📥 Downloading face landmark model (first time only)...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        _MODEL_PATH
    )
    print("✅ Face model downloaded")
else:
    print("✅ Face model found")


class FaceExpressionDetector:
    def __init__(self):
        # ── Load ML model if available ──
        self.model = None
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "models", "expression_model.pkl"
        )
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            print("✅ Expression ML model loaded")
        else:
            print("⚠️  No ML model — using 80 rule-based detectors")

        # ── Setup FaceLandmarker ──
        options = FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=VisionTaskRunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )
        self.face_landmarker = FaceLandmarker.create_from_options(options)
        print("✅ FaceExpressionDetector ready")

    # ─────────────────────────────────────
    #  LANDMARK WRAPPER CLASS
    # ─────────────────────────────────────
    class _LM:
        """Wraps mediapipe landmark to expose .x .y .z"""
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z

    # ─────────────────────────────────────
    #  MAIN DETECT
    # ─────────────────────────────────────
    def detect(self, frame):
        """
        Returns: (expression_label, confidence)
        """
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = self.face_landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None, 0.0

        # ── Draw landmarks on frame ──
        h, w, _ = frame.shape
        face_lm  = result.face_landmarks[0]
        for lm in face_lm:
            px = int(lm.x * w)
            py = int(lm.y * h)
            cv2.circle(frame, (px, py), 1, (0, 255, 150), -1)

        # ── Build compatible landmark list ──
        lm_list = [self._LM(lm.x, lm.y, lm.z) for lm in face_lm]

        if self.model:
            features = self._extract_features(lm_list)
            return self._ml_predict(features)
        else:
            return self._run_all_80_detectors(lm_list, h, w)

    # ─────────────────────────────────────
    #  HELPERS
    # ─────────────────────────────────────
    def _p(self, lm, idx, w, h):
        return np.array([lm[idx].x * w, lm[idx].y * h])

    def _dist(self, a, b):
        return np.linalg.norm(a - b)

    def _angle(self, a, b, c):
        ba  = a - b
        bc  = c - b
        cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return np.degrees(np.arccos(np.clip(cos, -1, 1)))

    # ─────────────────────────────────────
    #  80 DETECTORS
    # ─────────────────────────────────────
    def _run_all_80_detectors(self, lm, h, w):
        p      = lambda idx: self._p(lm, idx, w, h)
        d      = self._dist
        face_h = d(p(10), p(152)) + 1e-6
        scores = {}

        # ════════════════════════════════════
        # GROUP 1: MOUTH / LIP (1–20)
        # ════════════════════════════════════

        mouth_open  = d(p(13), p(14)) / face_h
        scores["open_mouth"]          = (mouth_open > 0.08, mouth_open * 5)

        mouth_width = d(p(61), p(291)) / face_h
        scores["wide_smile"]          = (mouth_width > 0.45, mouth_width * 2)

        lip_left_up = (p(10)[1] - p(61)[1]) / face_h
        scores["lip_corner_up_L"]     = (lip_left_up > 0.12, lip_left_up * 4)

        lip_right_up = (p(10)[1] - p(291)[1]) / face_h
        scores["lip_corner_up_R"]     = (lip_right_up > 0.12, lip_right_up * 4)

        lip_left_down = (p(61)[1] - p(10)[1]) / face_h
        scores["lip_corner_down_L"]   = (lip_left_down > 0.02, lip_left_down * 6)

        lip_right_down = (p(291)[1] - p(10)[1]) / face_h
        scores["lip_corner_down_R"]   = (lip_right_down > 0.02, lip_right_down * 6)

        upper_lip_raise = d(p(0), p(13)) / face_h
        scores["upper_lip_raised"]    = (upper_lip_raise < 0.10, (0.10 - upper_lip_raise) * 10)

        lower_lip_drop = d(p(17), p(14)) / face_h
        scores["lower_lip_dropped"]   = (lower_lip_drop > 0.06, lower_lip_drop * 8)

        lip_width  = d(p(61), p(291))
        lip_height = d(p(0),  p(17))
        lip_pucker = lip_height / (lip_width + 1e-6)
        scores["lip_pucker"]          = (lip_pucker > 0.8, lip_pucker)

        lip_press = d(p(13), p(14)) / face_h
        scores["lip_press"]           = (lip_press < 0.015, (0.015 - lip_press) * 50)

        upper_to_lower = p(13)[1] - p(14)[1]
        scores["lip_bite"]            = (upper_to_lower > 2, upper_to_lower / face_h * 5)

        mouth_left_stretch = p(61)[0] / w
        scores["mouth_left_stretch"]  = (mouth_left_stretch < 0.42, (0.42 - mouth_left_stretch) * 10)

        mouth_right_stretch = p(291)[0] / w
        scores["mouth_right_stretch"] = (mouth_right_stretch > 0.58, (mouth_right_stretch - 0.58) * 10)

        mouth_open_wide = d(p(13), p(14)) / face_h
        scores["mouth_open_wide"]     = (mouth_open_wide > 0.14, mouth_open_wide * 4)

        left_lift  = p(10)[1] - p(61)[1]
        right_lift = p(10)[1] - p(291)[1]
        symmetry   = 1 - abs(left_lift - right_lift) / (face_h + 1e-6)
        scores["smile_symmetry"]      = (symmetry > 0.92, symmetry)

        chin_drop = p(152)[1] / h
        scores["chin_drop"]           = (chin_drop > 0.82, (chin_drop - 0.82) * 10)

        jaw_width = d(p(234), p(454)) / face_h
        scores["jaw_clench"]          = (jaw_width < 0.55, (0.55 - jaw_width) * 8)

        asym_left = abs(p(61)[1] - p(291)[1]) / face_h
        scores["mouth_asym_left"]     = (asym_left > 0.03 and p(61)[1] < p(291)[1], asym_left * 10)
        scores["mouth_asym_right"]    = (asym_left > 0.03 and p(61)[1] > p(291)[1], asym_left * 10)

        lip_tremble = d(p(13), p(14)) / face_h
        scores["lip_tremble"]         = (0.01 < lip_tremble < 0.03, 0.03 - lip_tremble)

        # ════════════════════════════════════
        # GROUP 2: EYES (21–40)
        # ════════════════════════════════════

        left_eye_open  = d(p(159), p(145)) / face_h
        right_eye_open = d(p(386), p(374)) / face_h
        avg_eye        = (left_eye_open + right_eye_open) / 2

        scores["eyes_wide_open"]      = (avg_eye > 0.055, avg_eye * 10)
        scores["left_eye_wide"]       = (left_eye_open > 0.055, left_eye_open * 10)
        scores["right_eye_wide"]      = (right_eye_open > 0.055, right_eye_open * 10)
        scores["eyes_squeezed"]       = (avg_eye < 0.015, (0.015 - avg_eye) * 40)
        scores["left_eye_squint"]     = (left_eye_open < 0.022, (0.022 - left_eye_open) * 30)
        scores["right_eye_squint"]    = (right_eye_open < 0.022, (0.022 - right_eye_open) * 30)

        eye_asym = left_eye_open - right_eye_open
        scores["wink_left"]           = (eye_asym < -0.02, abs(eye_asym) * 20)
        scores["wink_right"]          = (eye_asym > 0.02,  abs(eye_asym) * 20)
        scores["eyes_normal"]         = (0.025 < avg_eye < 0.05, 0.6)
        scores["eyes_half_closed"]    = (0.015 < avg_eye < 0.025, (0.025 - avg_eye) * 20)

        left_lower_lid  = d(p(145), p(159)) / face_h
        right_lower_lid = d(p(374), p(386)) / face_h
        scores["left_eye_tear"]       = (left_lower_lid < 0.018, (0.018 - left_lower_lid) * 30)
        scores["right_eye_tear"]      = (right_lower_lid < 0.018, (0.018 - right_lower_lid) * 30)

        left_iris_y = lm[468].y if len(lm) > 468 else lm[159].y
        scores["eyes_looking_up"]     = (left_iris_y < 0.35, (0.35 - left_iris_y) * 5)
        scores["eyes_looking_down"]   = (left_iris_y > 0.55, (left_iris_y - 0.55) * 5)
        scores["blink_both"]          = (avg_eye < 0.012, (0.012 - avg_eye) * 60)
        scores["blink_left"]          = (left_eye_open < 0.012, (0.012 - left_eye_open) * 60)
        scores["blink_right"]         = (right_eye_open < 0.012, (0.012 - right_eye_open) * 60)

        left_brow_h = (p(33)[1] - p(70)[1]) / face_h
        eye_strain  = left_eye_open < 0.03 and left_brow_h < 0.06
        scores["eye_strain"]          = (eye_strain, 0.7 if eye_strain else 0)
        scores["surprised_eyes"]      = (avg_eye > 0.065, avg_eye * 8)
        scores["tired_eyes"]          = (0.012 < avg_eye < 0.022, (0.022 - avg_eye) * 25)

        # ════════════════════════════════════
        # GROUP 3: EYEBROWS (41–55)
        # ════════════════════════════════════

        left_brow_raise  = (p(33)[1]  - p(70)[1])  / face_h
        right_brow_raise = (p(362)[1] - p(336)[1]) / face_h
        avg_brow_raise   = (left_brow_raise + right_brow_raise) / 2

        scores["brows_raised"]        = (avg_brow_raise > 0.09, avg_brow_raise * 6)
        scores["left_brow_raised"]    = (left_brow_raise > 0.09, left_brow_raise * 6)
        scores["right_brow_raised"]   = (right_brow_raise > 0.09, right_brow_raise * 6)

        brow_dist = d(p(70), p(336)) / face_h
        scores["brows_furrowed"]      = (brow_dist < 0.28 and avg_brow_raise < 0.05,
                                         (0.05 - avg_brow_raise) * 10)
        scores["left_brow_down"]      = (left_brow_raise < 0.04, (0.04 - left_brow_raise) * 15)
        scores["right_brow_down"]     = (right_brow_raise < 0.04, (0.04 - right_brow_raise) * 15)

        brow_asym = left_brow_raise - right_brow_raise
        scores["brow_asym_left_up"]   = (brow_asym > 0.04,  abs(brow_asym) * 10)
        scores["brow_asym_right_up"]  = (brow_asym < -0.04, abs(brow_asym) * 10)

        inner_left  = (p(33)[1]  - p(55)[1])  / face_h
        inner_right = (p(362)[1] - p(285)[1]) / face_h
        scores["inner_brow_L"]        = (inner_left > 0.07, inner_left * 7)
        scores["inner_brow_R"]        = (inner_right > 0.07, inner_right * 7)

        outer_left  = (p(33)[1]  - p(46)[1])  / face_h
        outer_right = (p(362)[1] - p(276)[1]) / face_h
        scores["outer_brow_L"]        = (outer_left > 0.08, outer_left * 6)
        scores["outer_brow_R"]        = (outer_right > 0.08, outer_right * 6)

        brow_pinch = d(p(55), p(285)) / face_h
        scores["brow_pinch"]          = (brow_pinch < 0.22, (0.22 - brow_pinch) * 12)

        brow_flat = abs(avg_brow_raise - 0.065)
        scores["brow_flat"]           = (brow_flat < 0.01, 0.5)
        scores["dramatic_brow"]       = (avg_brow_raise > 0.13, avg_brow_raise * 5)

        # ════════════════════════════════════
        # GROUP 4: NOSE (56–62)
        # ════════════════════════════════════

        nose_wrinkle = d(p(1), p(2)) / face_h
        scores["nose_wrinkle"]        = (nose_wrinkle < 0.04, (0.04 - nose_wrinkle) * 15)

        nostril_l = d(p(64),  p(1)) / face_h
        nostril_r = d(p(294), p(1)) / face_h
        scores["nostril_flare_L"]     = (nostril_l > 0.07, nostril_l * 8)
        scores["nostril_flare_R"]     = (nostril_r > 0.07, nostril_r * 8)

        nose_shift = p(1)[0] / w
        scores["nose_tip_left"]       = (nose_shift < 0.47, (0.47 - nose_shift) * 10)
        scores["nose_tip_right"]      = (nose_shift > 0.53, (nose_shift - 0.53) * 10)

        bridge = d(p(6), p(197)) / face_h
        scores["nose_bridge_tens"]    = (bridge < 0.055, (0.055 - bridge) * 15)

        nose_down = p(1)[1] / h
        scores["nose_down"]           = (nose_down > 0.58, (nose_down - 0.58) * 8)

        # ════════════════════════════════════
        # GROUP 5: CHEEK & JAW (63–70)
        # ════════════════════════════════════

        cheek_l = p(116)[0] / w
        cheek_r = p(345)[0] / w
        scores["cheek_puff_L"]        = (cheek_l < 0.22, (0.22 - cheek_l) * 15)
        scores["cheek_puff_R"]        = (cheek_r > 0.78, (cheek_r - 0.78) * 15)

        cheek_raise_l = (p(61)[1]  - p(116)[1]) / face_h
        cheek_raise_r = (p(291)[1] - p(345)[1]) / face_h
        scores["cheek_raise_L"]       = (cheek_raise_l > 0.10, cheek_raise_l * 5)
        scores["cheek_raise_R"]       = (cheek_raise_r > 0.10, cheek_raise_r * 5)

        jaw_drop  = d(p(0), p(17)) / face_h
        scores["jaw_drop"]            = (jaw_drop > 0.22, jaw_drop * 3)

        jaw_shift = p(152)[0] / w
        scores["jaw_shift_L"]         = (jaw_shift < 0.46, (0.46 - jaw_shift) * 12)
        scores["jaw_shift_R"]         = (jaw_shift > 0.54, (jaw_shift - 0.54) * 12)

        face_tilt = (p(234)[1] - p(454)[1]) / face_h
        scores["face_tilt_L"]         = (face_tilt > 0.04, face_tilt * 8)

        # ════════════════════════════════════
        # GROUP 6: COMBINED EXPRESSIONS (71–80)
        # ════════════════════════════════════

        happy = (mouth_width > 0.42 and
                 (left_lift + right_lift) / 2 > face_h * 0.10 and
                 avg_eye > 0.02)
        scores["HAPPY"]               = (happy, 0.90 if happy else 0)

        sad = (lip_left_down > 0.02 and lip_right_down > 0.02 and
               inner_left > 0.05 and avg_eye < 0.035)
        scores["SAD"]                 = (sad, 0.88 if sad else 0)

        angry = (brow_dist < 0.30 and avg_brow_raise < 0.05 and
                 left_eye_open < 0.035 and lip_press < 0.02)
        scores["ANGRY"]               = (angry, 0.87 if angry else 0)

        surprised = (avg_brow_raise > 0.10 and avg_eye > 0.055 and mouth_open > 0.08)
        scores["SURPRISED"]           = (surprised, 0.92 if surprised else 0)

        pain = (brow_pinch < 0.24 and avg_eye < 0.03 and mouth_open > 0.05)
        scores["PAIN"]                = (pain, 0.85 if pain else 0)

        disgust = (nose_wrinkle < 0.042 and upper_lip_raise < 0.10 and
                   avg_brow_raise < 0.06)
        scores["DISGUST"]             = (disgust, 0.82 if disgust else 0)

        fear = (avg_brow_raise > 0.09 and avg_eye > 0.05 and
                0.02 < mouth_open < 0.08)
        scores["FEAR"]                = (fear, 0.80 if fear else 0)

        contempt = (asym_left > 0.025 and abs(brow_asym) > 0.03)
        scores["CONTEMPT"]            = (contempt, 0.75 if contempt else 0)

        neutral = (0.025 < avg_eye < 0.05 and
                   0.04 < avg_brow_raise < 0.09 and
                   mouth_open < 0.06 and
                   mouth_width < 0.42)
        scores["NEUTRAL"]             = (neutral, 0.65 if neutral else 0)

        tired = (0.012 < avg_eye < 0.025 and
                 avg_brow_raise < 0.065 and
                 lip_left_down > 0.01)
        scores["TIRED"]               = (tired, 0.72 if tired else 0)

        # ─────────────────────────────────────
        #  PICK BEST MATCH
        # ─────────────────────────────────────
        priority_labels = [
            "SURPRISED", "HAPPY", "ANGRY", "SAD", "PAIN",
            "DISGUST", "FEAR", "CONTEMPT", "TIRED", "NEUTRAL"
        ]

        for label in priority_labels:
            detected, conf = scores[label]
            if detected and conf > 0.5:
                return label.lower(), conf

        best_label = "neutral"
        best_score = 0.0
        for label, (detected, score) in scores.items():
            if detected and score > best_score and label not in priority_labels:
                best_score = score
                best_label = label

        if best_score > 0.3:
            return best_label.replace("_", " "), min(best_score, 0.99)

        return "neutral", 0.65

    # ─────────────────────────────────────
    #  FEATURE EXTRACTION
    # ─────────────────────────────────────
    def _extract_features(self, landmarks):
        LIP_LANDMARKS = [61, 146, 91, 181, 84, 17, 314, 405,
                         321, 375, 291, 308, 324, 318, 402,
                         317, 14, 87, 178, 88, 95]
        features = []
        for idx in LIP_LANDMARKS:
            lm = landmarks[idx]
            features.extend([lm.x, lm.y, lm.z])
        return np.array(features, dtype=np.float32)

    def _ml_predict(self, features):
        try:
            pred  = self.model.predict([features])[0]
            proba = self.model.predict_proba([features])[0]
            return pred, float(np.max(proba))
        except Exception:
            return None, 0.0

    def get_all_landmarks(self, frame):
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = self.face_landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None
        return [(lm.x, lm.y, lm.z) for lm in result.face_landmarks[0]]