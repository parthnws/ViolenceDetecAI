# AI Violence Detection – Video Surveillance

Python + OpenCV + TensorFlow + Streamlit.

## Features
- Real-time analysis of uploaded video, webcam, or CCTV (RTSP URL)
- Violence score from a MobileNetV2 + LSTM model (16-frame clips)
- Weapon identification (gun / knife / bat, etc.) via a YOLO weapon model
- 3 severity levels, adjustable thresholds
  - Level 1 Low (score ≥ 0.50) · Level 2 Medium (≥ 0.70) · Level 3 High (≥ 0.88)
  - Weapons escalate severity: firearm → Level 3, knife → at least Level 2, blunt object → at least Level 1
- Live score chart, incident log, emergency contacts panel (India: 112, 100, 102/108, 101, 1091/181, 1098, 1930)

## Setup
```
pip install -r requirements.txt scikit-learn
```

## 1. Train the violence model
Download a dataset (e.g. RWF-2000, or "Real Life Violence Situations" on Kaggle) into:
```
data/Violence/   data/NonViolence/
python train_model.py --data data --epochs 15
```
This writes `models/violence_model.keras`.

## 2. Weapon model (optional)
Train YOLOv8 on a weapon dataset (Roboflow Universe has several) and save the weights as `models/weapon.pt`.
Class names containing gun/pistol/rifle/knife/bat etc. are mapped to danger weights in `detector.py`.

## 3. Run
```
streamlit run app.py
```
Without trained models the app runs in **demo mode** (optical-flow motion heuristic only, not a real classifier).

## Deploy on Streamlit Community Cloud
1. Push this folder to GitHub (use `opencv-python-headless`; keep model files < 100 MB or use Git LFS).
2. share.streamlit.io → New app → select repo → main file `app.py`.
3. Note: cloud servers can't open your local webcam; use uploaded videos or an RTSP/HTTP stream URL.

## Notes
- Accuracy depends on training data; test on your own camera footage and tune thresholds.
- This is a decision-support tool. Always have a human verify alerts before acting.
