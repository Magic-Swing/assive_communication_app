"""
train_model.py
═══════════════════════════════════════════════════════════
  ML MODEL TRAINER
  Trains gesture & expression classifiers using
  Random Forest + SVM (scikit-learn)

  HOW TO USE:
    1. First collect data using collect_data.py
    2. Run: python train_model.py
    3. Models saved to models/ folder automatically
    4. Restart main_app.py to use new models
═══════════════════════════════════════════════════════════
"""

import os
import csv
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score


# ─────────────────────────────────────────
#  LOAD DATASET FROM CSV FILES
# ─────────────────────────────────────────
def load_dataset(data_folder):
    """
    Loads all CSV files from data_folder subfolders.
    Each CSV file = one sample.
    Last column = label.
    Returns: X (features), y (labels)
    """
    X, y = [], []

    if not os.path.exists(data_folder):
        print(f"❌ Folder not found: {data_folder}")
        return None, None

    for label in os.listdir(data_folder):
        label_folder = os.path.join(data_folder, label)
        if not os.path.isdir(label_folder):
            continue

        files = [f for f in os.listdir(label_folder) if f.endswith(".csv")]
        if not files:
            continue

        print(f"  Loading '{label}': {len(files)} samples")

        for fname in files:
            fpath = os.path.join(label_folder, fname)
            try:
                with open(fpath, "r") as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if not row:
                            continue
                        # Last item is label, rest are features
                        features = [float(v) for v in row[:-1]]
                        file_label = row[-1].strip()
                        X.append(features)
                        y.append(file_label)
            except Exception as e:
                print(f"  ⚠️  Error reading {fname}: {e}")

    if not X:
        return None, None

    return np.array(X), np.array(y)


# ─────────────────────────────────────────
#  TRAIN MODEL
# ─────────────────────────────────────────
def train(data_folder, model_name):
    print(f"\n{'='*50}")
    print(f"  Training: {model_name}")
    print(f"  Data: {data_folder}")
    print(f"{'='*50}")

    # Load data
    X, y = load_dataset(data_folder)

    if X is None or len(X) == 0:
        print(f"  ❌ No data found in {data_folder}")
        print(f"  Run collect_data.py first!\n")
        return

    print(f"\n  Total samples : {len(X)}")
    print(f"  Feature size  : {X.shape[1]}")
    print(f"  Classes       : {list(np.unique(y))}")

    # Check minimum samples
    unique, counts = np.unique(y, return_counts=True)
    for label, count in zip(unique, counts):
        status = "✅" if count >= 30 else "⚠️  (need more samples!)"
        print(f"    {status} '{label}': {count} samples")

    if len(X) < 20:
        print("\n  ❌ Not enough data. Collect at least 30 samples per gesture.")
        return

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(unique) > 1 else None
    )

    print(f"\n  Train: {len(X_train)} | Test: {len(X_test)}")

    # ── Train Random Forest ──
    print("\n  Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf_model.predict(X_test))
    print(f"  Random Forest Accuracy: {rf_acc:.2%}")

    # ── Train SVM ──
    print("\n  Training SVM...")
    svm_model = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        probability=True,
        random_state=42
    )
    svm_model.fit(X_train, y_train)
    svm_acc = accuracy_score(y_test, svm_model.predict(X_test))
    print(f"  SVM Accuracy: {svm_acc:.2%}")

    # ── Pick best model ──
    if rf_acc >= svm_acc:
        best_model = rf_model
        best_name  = "Random Forest"
        best_acc   = rf_acc
    else:
        best_model = svm_model
        best_name  = "SVM"
        best_acc   = svm_acc

    print(f"\n  🏆 Best model: {best_name} ({best_acc:.2%})")

    # ── Print classification report ──
    print("\n  Classification Report:")
    print(classification_report(y_test, best_model.predict(X_test)))

    # ── Save model ──
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", f"{model_name}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)

    print(f"  ✅ Model saved: {model_path}")
    print(f"  Restart main_app.py to use the new model.\n")


# ─────────────────────────────────────────
if __name__ == "__main__":
    print("\n╔══════════════════════════════════════╗")
    print("║    ASSISTIVE APP - MODEL TRAINER     ║")
    print("╚══════════════════════════════════════╝")

    # Train gesture model
    train(
        data_folder=os.path.join("data", "gestures"),
        model_name="gesture_model"
    )

    # Train expression model
    train(
        data_folder=os.path.join("data", "expressions"),
        model_name="expression_model"
    )

    print("🎉 Training complete!\n")
