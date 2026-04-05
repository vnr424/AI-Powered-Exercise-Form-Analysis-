Exercise Form Analysis System
Real-Time Exercise Form Analysis Using MediaPipe and Random Forest

Requirements:
- Python 3.12
- See requirements.txt for dependencies

Installation:
pip install -r requirements.txt

Usage:

Desktop Application:
python scripts/realtime_with_medical_feedback.py

Web Backend (for web interface):
python scripts/websocket_server.py

Training:
python scripts/retrain_combined_model.py

Models:
- Random Forest: models/random_forest_tuned.joblib
- Feature Scaler: models/feature_scaler_tuned.joblib
- MediaPipe Pose: models/pose_landmarker_heavy.task

Data:
- Training data: data/combined_training_data.csv (5,804 samples)
- Shoulder press: data/shoulder_press_features.csv (1,022 samples)
