"""
voice_output.py
─────────────────────────────────────────
Text-to-Speech engine for the Assistive App
Non-blocking queue-based speech output
─────────────────────────────────────────
"""

import threading
import queue
import os


class VoiceEngine:
    def __init__(self):
        self.engine = None
        self._queue = queue.Queue()
        self._speaking = False
        self._init_engine()
        # Start background worker thread
        self._worker = threading.Thread(target=self._speech_worker, daemon=True)
        self._worker.start()

    def _init_engine(self):
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", 160)    # Slightly faster
            self.engine.setProperty("volume", 1.0)

            voices = self.engine.getProperty("voices")
            for voice in voices:
                if "english" in voice.name.lower():
                    self.engine.setProperty("voice", voice.id)
                    break

            print("✅ Voice engine initialized (pyttsx3)")
        except Exception as e:
            print(f"⚠️  pyttsx3 failed: {e}")
            self.engine = None

    def _speech_worker(self):
        """Background thread — processes speech queue one at a time"""
        while True:
            text = self._queue.get()
            if text is None:
                break
            self._speaking = True
            try:
                if self.engine:
                    self.engine.say(text)
                    self.engine.runAndWait()
            except Exception as e:
                print(f"Speech error: {e}")
                # Reinitialize engine if it crashes
                try:
                    self._init_engine()
                except:
                    pass
            finally:
                self._speaking = False
            self._queue.task_done()

    def speak(self, text):
        """Add text to speech queue — non-blocking"""
        if not text:
            return
        # Clear old queued items so latest message plays immediately
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break
        self._queue.put(text)

    def is_speaking(self):
        return self._speaking

    def set_rate(self, rate):
        if self.engine:
            self.engine.setProperty("rate", rate)

    def set_volume(self, volume):
        if self.engine:
            self.engine.setProperty("volume", volume)

    def stop(self):
        self._queue.put(None)