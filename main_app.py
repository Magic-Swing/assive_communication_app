"""
========================================
  ASSISTIVE COMMUNICATION APP
  For people who cannot speak, hear, or see
  Detects Face Expressions + Hand Gestures
  Outputs: Text + Voice
========================================
"""

import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import tkinter as tk
from tkinter import ttk, font
import threading
import time
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.face_detector import FaceExpressionDetector
from utils.hand_detector import HandGestureDetector
from utils.voice_output import VoiceEngine
from utils.model_loader import ModelLoader


# ─────────────────────────────────────────
#  MAIN APPLICATION CLASS
# ─────────────────────────────────────────
class AssistiveApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Assistive Communication App")
        self.root.geometry("1100x700")
        self.root.configure(bg="#1a1a2e")
        self.root.resizable(True, True)

        # State
        self.running = False
        self.cap = None
        self.last_spoken = ""
        self.last_spoken_time = 0
        self.speak_cooldown = 4       # seconds before repeating SAME message
        self.frame_skip = 0           # frame counter for skipping
        self.frame_process_every = 3  # process every 3rd frame for speed

        # Detectors
        self.face_detector = FaceExpressionDetector()
        self.hand_detector = HandGestureDetector()
        self.voice_engine = VoiceEngine()
        self.model_loader = ModelLoader()

        self._build_ui()

    # ─────────────────────────────────────
    #  BUILD UI
    # ─────────────────────────────────────
    def _build_ui(self):
        # ── Title Bar ──
        title_frame = tk.Frame(self.root, bg="#16213e", pady=10)
        title_frame.pack(fill="x")

        tk.Label(
            title_frame,
            text="🤝  Assistive Communication App",
            font=("Helvetica", 22, "bold"),
            fg="#e94560",
            bg="#16213e"
        ).pack()

        tk.Label(
            title_frame,
            text="Helping people communicate through expressions & gestures",
            font=("Helvetica", 11),
            fg="#a8a8b3",
            bg="#16213e"
        ).pack()

        # ── Main Content ──
        main_frame = tk.Frame(self.root, bg="#1a1a2e")
        main_frame.pack(fill="both", expand=True, padx=15, pady=10)

        # Left: Camera Feed
        left_frame = tk.Frame(main_frame, bg="#1a1a2e")
        left_frame.pack(side="left", fill="both", expand=True)

        tk.Label(
            left_frame,
            text="📷  Live Camera Feed",
            font=("Helvetica", 13, "bold"),
            fg="#ffffff",
            bg="#1a1a2e"
        ).pack(pady=(0, 5))

        self.camera_label = tk.Label(
            left_frame,
            bg="#0f3460",
            text="Camera feed will appear here\nClick 'Start Camera'",
            fg="#a8a8b3",
            font=("Helvetica", 13),
            width=60,
            height=20
        )
        self.camera_label.pack(fill="both", expand=True)

        # Right: Detection Panel
        right_frame = tk.Frame(main_frame, bg="#1a1a2e", width=320)
        right_frame.pack(side="right", fill="y", padx=(15, 0))
        right_frame.pack_propagate(False)

        # ── Detection Results ──
        tk.Label(
            right_frame,
            text="🧠  Detection Results",
            font=("Helvetica", 13, "bold"),
            fg="#ffffff",
            bg="#1a1a2e"
        ).pack(pady=(0, 8))

        # Expression Box
        expr_card = tk.Frame(right_frame, bg="#16213e", pady=10, padx=10)
        expr_card.pack(fill="x", pady=5)
        tk.Label(expr_card, text="😊 Face Expression", font=("Helvetica", 10, "bold"),
                 fg="#e94560", bg="#16213e").pack(anchor="w")
        self.expression_var = tk.StringVar(value="Waiting...")
        tk.Label(expr_card, textvariable=self.expression_var,
                 font=("Helvetica", 18, "bold"), fg="#ffffff", bg="#16213e").pack(pady=5)

        # Gesture Box
        gest_card = tk.Frame(right_frame, bg="#16213e", pady=10, padx=10)
        gest_card.pack(fill="x", pady=5)
        tk.Label(gest_card, text="✋ Hand Gesture", font=("Helvetica", 10, "bold"),
                 fg="#0f3460", bg="#16213e").pack(anchor="w")
        self.gesture_var = tk.StringVar(value="Waiting...")
        tk.Label(gest_card, textvariable=self.gesture_var,
                 font=("Helvetica", 18, "bold"), fg="#ffffff", bg="#16213e").pack(pady=5)

        # ── Message Output ──
        tk.Label(
            right_frame,
            text="💬  Message Output",
            font=("Helvetica", 13, "bold"),
            fg="#ffffff",
            bg="#1a1a2e"
        ).pack(pady=(15, 5))

        self.message_box = tk.Text(
            right_frame,
            height=7,
            font=("Helvetica", 13),
            bg="#0f3460",
            fg="#00ff88",
            wrap="word",
            bd=0,
            padx=10,
            pady=10
        )
        self.message_box.pack(fill="x")
        self.message_box.insert("end", "Messages will appear here...\n")
        self.message_box.config(state="disabled")

        # ── Controls ──
        ctrl_frame = tk.Frame(right_frame, bg="#1a1a2e")
        ctrl_frame.pack(fill="x", pady=10)

        self.start_btn = tk.Button(
            ctrl_frame,
            text="▶  Start Camera",
            font=("Helvetica", 11, "bold"),
            bg="#00ff88",
            fg="#1a1a2e",
            relief="flat",
            padx=10, pady=8,
            cursor="hand2",
            command=self.start_camera
        )
        self.start_btn.pack(fill="x", pady=3)

        self.stop_btn = tk.Button(
            ctrl_frame,
            text="⏹  Stop Camera",
            font=("Helvetica", 11, "bold"),
            bg="#e94560",
            fg="#ffffff",
            relief="flat",
            padx=10, pady=8,
            cursor="hand2",
            state="disabled",
            command=self.stop_camera
        )
        self.stop_btn.pack(fill="x", pady=3)

        tk.Button(
            ctrl_frame,
            text="🔊  Speak Again",
            font=("Helvetica", 11, "bold"),
            bg="#0f3460",
            fg="#ffffff",
            relief="flat",
            padx=10, pady=8,
            cursor="hand2",
            command=self.speak_again
        ).pack(fill="x", pady=3)

        tk.Button(
            ctrl_frame,
            text="🗑  Clear Messages",
            font=("Helvetica", 11, "bold"),
            bg="#333355",
            fg="#ffffff",
            relief="flat",
            padx=10, pady=8,
            cursor="hand2",
            command=self.clear_messages
        ).pack(fill="x", pady=3)

        # ── Status Bar ──
        self.status_var = tk.StringVar(value="⚪  Ready — Click 'Start Camera' to begin")
        tk.Label(
            self.root,
            textvariable=self.status_var,
            font=("Helvetica", 10),
            fg="#a8a8b3",
            bg="#16213e",
            anchor="w",
            padx=10
        ).pack(fill="x", side="bottom")

    # ─────────────────────────────────────
    #  CAMERA CONTROL
    # ─────────────────────────────────────
    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_var.set("❌  Camera not found! Check your webcam.")
            return

        self.running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_var.set("🟢  Camera running — detecting expressions & gestures...")

        thread = threading.Thread(target=self._camera_loop, daemon=True)
        thread.start()

    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.camera_label.config(image="", text="Camera stopped\nClick 'Start Camera' to resume")
        self.status_var.set("⚪  Camera stopped")

    # ─────────────────────────────────────
    #  MAIN CAMERA LOOP (runs in thread)
    # ─────────────────────────────────────
    def _camera_loop(self):
        from PIL import Image, ImageTk

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # Mirror
            display_frame = frame.copy()

            # ── Skip frames for speed (only process every Nth frame) ──
            self.frame_skip += 1
            if self.frame_skip % self.frame_process_every == 0:

                # ── Detect Hand Gesture (priority) ──
                gesture, gesture_confidence, annotated = self.hand_detector.detect(display_frame)
                display_frame = annotated
                if gesture:
                    self.gesture_var.set(f"{gesture}")
                    cv2.putText(display_frame, f"Gesture: {gesture} ({gesture_confidence:.0%})",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                else:
                    self.gesture_var.set("None")

                # ── Detect Face Expression ──
                expression, face_confidence = self.face_detector.detect(frame)
                if expression:
                    self.expression_var.set(f"{expression}")
                    cv2.putText(display_frame, f"Expression: {expression} ({face_confidence:.0%})",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 150), 2)
                else:
                    self.expression_var.set("None")

                # ── Generate & Output Message ──
                message = self._generate_message(expression, gesture)
                if message:
                    self._output_message(message)

            # ── Show frame in UI ──
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            img = img.resize((640, 480), Image.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            self.camera_label.config(image=imgtk, text="")
            self.camera_label.image = imgtk

        self.cap.release()

    # ─────────────────────────────────────
    #  MESSAGE GENERATION
    # ─────────────────────────────────────
    def _generate_message(self, expression, gesture):
        """Map detected expression/gesture to a meaningful message"""

        # Gesture takes priority over expression
        gesture_messages = {
            "help":  "I need HELP! Please come quickly!",
            "water": "I am thirsty. I need water please.",
            "food":  "I am hungry. I need food please.",
            "yes":   "Yes, that is correct.",
            "no":    "No, that is not what I want.",
            "pain":  "I am in PAIN. Please help me.",
        }

        expression_messages = {
            "happy":     "I am feeling happy and good.",
            "sad":       "I am feeling sad. Please talk to me.",
            "angry":     "I am upset or uncomfortable.",
            "pain":      "I am in pain. I need medical attention.",
            "surprised": "I am surprised!",
            "tired":     "I am feeling very tired.",
            "disgust":   "I am feeling disgusted or uncomfortable.",
            "fear":      "I am feeling scared or anxious.",
        }

        if gesture and gesture in gesture_messages:
            return gesture_messages[gesture]
        if expression and expression in expression_messages:
            return expression_messages[expression]
        return None

    def _output_message(self, message):
        """Display and speak message with smart cooldown"""
        now = time.time()

        # Always speak if it's a NEW/different message
        # Only apply cooldown if SAME message is repeating
        if message == self.last_spoken and (now - self.last_spoken_time) < self.speak_cooldown:
            return

        self.last_spoken = message
        self.last_spoken_time = now

        # Update text box
        self.message_box.config(state="normal")
        timestamp = time.strftime("%H:%M:%S")
        self.message_box.insert("end", f"[{timestamp}] {message}\n")
        self.message_box.see("end")
        self.message_box.config(state="disabled")

        # Speak immediately (non-blocking queue)
        self.voice_engine.speak(message)

    # ─────────────────────────────────────
    #  BUTTON ACTIONS
    # ─────────────────────────────────────
    def speak_again(self):
        if self.last_spoken:
            self.voice_engine.speak(self.last_spoken)

    def clear_messages(self):
        self.message_box.config(state="normal")
        self.message_box.delete("1.0", "end")
        self.message_box.config(state="disabled")

    # ─────────────────────────────────────
    #  RUN APP
    # ─────────────────────────────────────
    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    def _on_close(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.voice_engine.stop()
        self.root.destroy()


# ─────────────────────────────────────────
if __name__ == "__main__":
    print("Starting Assistive Communication App...")
    app = AssistiveApp()
    app.run()