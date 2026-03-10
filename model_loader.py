"""
model_loader.py
─────────────────────────────────────────
Loads trained ML models for gesture & expression
─────────────────────────────────────────
"""
import os
import pickle


class ModelLoader:
    def __init__(self):
        self.models_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "models"
        )

    def load(self, name):
        path = os.path.join(self.models_dir, f"{name}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        return None

    def save(self, model, name):
        os.makedirs(self.models_dir, exist_ok=True)
        path = os.path.join(self.models_dir, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(model, f)
        print(f"✅ Model saved: {path}")
