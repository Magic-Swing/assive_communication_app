"""
collect_data.py
═══════════════════════════════════════════════════════════
  DATA COLLECTION TOOL
  Collects hand gesture & face expression landmark data
  to train your ML model
═══════════════════════════════════════════════════════════

HOW TO USE:
  1. Run: python collect_data.py
  2. Select mode: gesture or expression
  3. Enter label name (e.g. "help", "water", "happy")
  4. Show gesture/expression to camera
  5. Press 'S' to save each sample
  6. Collect 200+ samples per gesture
  7. Repeat for each gesture/expression label
═══════════════════════════════════════════════════════════
"""

import cv2
import mediapipe as mp
import csv
import os
import time


# ─────────────────────────────────────────
#  SETTINGS — change these before running
# ─────────────────────────────────────────
MODE           = "gesture"    # "gesture" or "expression"
LABEL_NAME     = "up"       # name of gesture/expression
SAMPLES_TARGET = 30        # how many samples to collect
AUTO_CAPTURE   = False        # True = auto save every 0.5s
AUTO_INTERVAL  = 0.5          # seconds between auto captures      # True = auto save every 0.5s    # seconds between auto captures
# ─────────────────────────────────────────


def collect_gesture_data():
    """Collect hand landmark data (21 points × 3 = 63 values)"""

    mp_hands   = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands      = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6
    )

    save_folder = os.path.join("data", "gestures", LABEL_NAME)
    os.makedirs(save_folder, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0
    last_capture_time = 0

    print(f"\n{'='*50}")
    print(f"  Collecting GESTURE data for: '{LABEL_NAME}'")
    print(f"  Target: {SAMPLES_TARGET} samples")
    print(f"  Press 'S' to save | 'Q' to quit")
    print(f"{'='*50}\n")

    while count < SAMPLES_TARGET:
        ret, frame = cap.read()
        if not ret:
            break

        frame     = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = hands.process(rgb_frame)

        saved_this_frame = False

        if results.multi_hand_landmarks:
            hand_lm = results.multi_hand_landmarks[0]

            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

            # Extract 21 × 3 = 63 values
            row  = []
            wrist = hand_lm.landmark[0]
            for lm in hand_lm.landmark:
                row.extend([
                    lm.x - wrist.x,  # normalized x
                    lm.y - wrist.y,  # normalized y
                    lm.z - wrist.z   # normalized z
                ])
            row.append(LABEL_NAME)  # label at end

            # Save on 'S' key or auto
            key = cv2.waitKey(1) & 0xFF
            now = time.time()

            should_save = (key == ord('s'))
            if AUTO_CAPTURE and (now - last_capture_time) >= AUTO_INTERVAL:
                should_save = True

            if should_save:
                filepath = os.path.join(save_folder, f"sample_{count:04d}.csv")
                with open(filepath, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
                count += 1
                last_capture_time = now
                print(f"  ✅ Saved sample {count:3d}/{SAMPLES_TARGET}")
                saved_this_frame = True

            if key == ord('q'):
                break

        # ── UI Overlay ──
        color = (0, 255, 100) if results.multi_hand_landmarks else (0, 100, 255)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 90), (20, 20, 40), -1)
        cv2.putText(frame, f"Gesture: {LABEL_NAME}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f"Saved: {count}/{SAMPLES_TARGET}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        status = "HAND DETECTED — Press S to save" if results.multi_hand_landmarks else "No hand detected"
        cv2.putText(frame, status, (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Collect Gesture Data", frame)
        if not saved_this_frame:
            cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Done! Collected {count} samples for '{LABEL_NAME}'")
    print(f"   Saved to: data/gestures/{LABEL_NAME}/\n")


def collect_expression_data():
    """Collect face landmark data (key 63 values from 21 lip points)"""

    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing   = mp.solutions.drawing_utils
    mp_styles    = mp.solutions.drawing_styles
    face_mesh    = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )

    # Key lip landmarks for expression
    LIP_LANDMARKS = [61, 146, 91, 181, 84, 17, 314, 405,
                     321, 375, 291, 308, 324, 318, 402,
                     317, 14, 87, 178, 88, 95]

    save_folder = os.path.join("data", "expressions", LABEL_NAME)
    os.makedirs(save_folder, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0
    last_capture_time = 0

    print(f"\n{'='*50}")
    print(f"  Collecting EXPRESSION data for: '{LABEL_NAME}'")
    print(f"  Target: {SAMPLES_TARGET} samples")
    print(f"  Press 'S' to save | 'Q' to quit")
    print(f"{'='*50}\n")

    while count < SAMPLES_TARGET:
        ret, frame = cap.read()
        if not ret:
            break

        frame   = cv2.flip(frame, 1)
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        saved_this_frame = False

        if results.multi_face_landmarks:
            face_lm = results.multi_face_landmarks[0]

            mp_drawing.draw_landmarks(
                frame, face_lm,
                mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style()
            )

            # Extract key landmarks
            row = []
            for idx in LIP_LANDMARKS:
                lm = face_lm.landmark[idx]
                row.extend([lm.x, lm.y, lm.z])
            row.append(LABEL_NAME)

            key = cv2.waitKey(1) & 0xFF
            now = time.time()

            should_save = (key == ord('s'))
            if AUTO_CAPTURE and (now - last_capture_time) >= AUTO_INTERVAL:
                should_save = True

            if should_save:
                filepath = os.path.join(save_folder, f"sample_{count:04d}.csv")
                with open(filepath, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
                count += 1
                last_capture_time = now
                print(f"  ✅ Saved sample {count:3d}/{SAMPLES_TARGET}")
                saved_this_frame = True

            if key == ord('q'):
                break

        color = (0, 255, 100) if results.multi_face_landmarks else (0, 100, 255)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 90), (20, 20, 40), -1)
        cv2.putText(frame, f"Expression: {LABEL_NAME}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f"Saved: {count}/{SAMPLES_TARGET}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        status = "FACE DETECTED — Press S to save" if results.multi_face_landmarks else "No face detected"
        cv2.putText(frame, status, (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Collect Expression Data", frame)
        if not saved_this_frame:
            cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Done! Collected {count} samples for '{LABEL_NAME}'")
    print(f"   Saved to: data/expressions/{LABEL_NAME}/\n")


# ─────────────────────────────────────────
if __name__ == "__main__":
    print("\n╔══════════════════════════════════════╗")
    print("║    ASSISTIVE APP - DATA COLLECTOR    ║")
    print("╚══════════════════════════════════════╝")
    print(f"\n  MODE  : {MODE}")
    print(f"  LABEL : {LABEL_NAME}")
    print(f"  TARGET: {SAMPLES_TARGET} samples")
    print(f"\n  Edit MODE and LABEL_NAME at top of file to change.\n")

    if MODE == "gesture":
        collect_gesture_data()
    elif MODE == "expression":
        collect_expression_data()
    else:
        print("❌ Invalid MODE. Set to 'gesture' or 'expression'")
