# 🤝 Assistive Communication App
### For people who cannot speak, hear, or see

This app uses your **webcam** to detect face expressions and hand gestures,
then speaks out loud what the person is trying to communicate.

---

## 📁 Folder Structure

```
assistive_app/
│
├── main_app.py              ← 🚀 Run this to start the app
├── collect_data.py          ← 📷 Step 1: Collect gesture/expression data
├── train_model.py           ← 🧠 Step 2: Train the ML model
├── requirements.txt         ← 📦 Install dependencies
│
├── utils/
│   ├── face_detector.py     ← FaceMesh 468 landmarks detector
│   ├── hand_detector.py     ← Hand 21 landmarks detector
│   ├── voice_output.py      ← Text-to-Speech engine
│   └── model_loader.py      ← Load/save trained models
│
├── data/
│   ├── gestures/
│   │   ├── help/            ← CSV files for "help" gesture
│   │   ├── water/           ← CSV files for "water" gesture
│   │   ├── food/            ← CSV files for "food" gesture
│   │   ├── yes/             ← CSV files for "yes" gesture
│   │   ├── no/              ← CSV files for "no" gesture
│   │   └── pain/            ← CSV files for "pain" gesture
│   └── expressions/
│       ├── happy/
│       ├── sad/
│       ├── angry/
│       ├── pain/
│       └── neutral/
│
└── models/
    ├── gesture_model.pkl    ← Saved gesture ML model (after training)
    └── expression_model.pkl ← Saved expression ML model (after training)
```

---

## ⚙️ Setup (Step by Step)

### Step 1 — Install Python Libraries
Open terminal in PyCharm and run:
```bash
pip install -r requirements.txt
```

### Step 2 — Run the App (works immediately!)
```bash
python main_app.py
```
> The app works right away using **rule-based detection** — no training needed yet!

---

## 🎓 Improve Accuracy with ML Training

### Step 3 — Collect Gesture Data
Open `collect_data.py` and set:
```python
MODE       = "gesture"   # or "expression"
LABEL_NAME = "help"      # name of gesture
```
Then run:
```bash
python collect_data.py
```
- Show the gesture to the camera
- Press **S** to save each sample
- Collect **200+ samples** per gesture
- Repeat for: help, water, food, yes, no, pain

### Step 4 — Train the Model
```bash
python train_model.py
```
Models are automatically saved to `models/` folder.

### Step 5 — Restart App
```bash
python main_app.py
```
The app now uses your trained ML model for better accuracy!

---

## 🤟 Supported Gestures & Their Meanings

| Gesture | Meaning |
|---------|---------|
| ✋ Open Palm | "I need HELP! Please come!" |
| 👍 Thumbs Up | "Yes, that is correct." |
| ✌️ Peace Sign | "I need food please." |
| ☝️ Index Finger | "I need water please." |
| ✊ Fist | "I am in pain." |
| 🤙 Shaka | "Call for help." |

## 😊 Detected Expressions & Meanings

| Expression | Meaning |
|------------|---------|
| Happy | "I am feeling happy." |
| Sad | "I am feeling sad." |
| Angry | "I am upset or uncomfortable." |
| Surprised | "I am surprised!" |
| Pain | "I am in pain. I need help." |
| Neutral | (no message spoken) |

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python 3.8+ | Main language |
| OpenCV | Camera & video processing |
| MediaPipe FaceMesh | 468 face landmark detection |
| MediaPipe Hands | 21 hand landmark detection |
| scikit-learn | Random Forest + SVM ML models |
| pyttsx3 | Offline text-to-speech |
| Tkinter | User Interface |

---

## 📱 Future: Android Deployment
1. Convert UI to **Kivy** framework
2. Use **Buildozer** to create APK
3. Or use **Flask API** + Android frontend

---

Made with ❤️ to help people communicate.
