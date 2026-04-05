import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from pathlib import Path

print("="*60)
print("RETRAINING MODEL WITH BENCH PRESS + SHOULDER PRESS")
print("="*60)

# Feature columns (18 features)
feature_columns = [
    'left_elbow_angle', 'right_elbow_angle', 'avg_elbow_angle',
    'elbow_angle_diff', 'elbow_symmetry',
    'wrist_x', 'wrist_y', 'wrist_z', 'wrist_x_normalized', 'wrist_y_position',
    'shoulder_width', 'shoulder_mid_x', 'retraction_offset', 'retraction_normalized',
    'alignment_score', 'has_flare', 'extreme_flare', 'avg_elbow_angle_squared'
]

# Load existing bench press data (JSON)
print("\n1. Loading existing bench press data...")
json_file = Path('../data/biomechanical_features_augmented.json')

with open(json_file, 'r') as f:
    json_data = json.load(f)

bench_data = json_data['data']  # Access the nested 'data' key

# Convert to DataFrame
bench_rows = []
for item in bench_data:
    row = {}
    # Map features
    for feat in feature_columns:
        if feat in item:
            row[feat] = item[feat]
        elif feat == 'elbow_angle_diff' and 'elbow_asymmetry' in item:
            row[feat] = item['elbow_asymmetry']
        elif feat == 'has_flare' and 'excessive_elbow_flare' in item:
            row[feat] = item['excessive_elbow_flare']
        elif feat == 'extreme_flare' and 'extreme_elbow_flare' in item:
            row[feat] = item['extreme_elbow_flare']
    
    # Convert label
    label_val = item.get('label', 'correct')
    row['label'] = 1 if label_val == 'correct' else 0
    
    # Only add if we have most features
    if len(row) >= 15:  # At least 15 of 18 features
        bench_rows.append(row)

df_bench = pd.DataFrame(bench_rows)

# Fill missing columns with 0 if needed
for feat in feature_columns:
    if feat not in df_bench.columns:
        df_bench[feat] = 0

print(f"    Bench press samples: {len(df_bench)}")
print(f"      Correct: {len(df_bench[df_bench['label']==1])}")
print(f"      Incorrect: {len(df_bench[df_bench['label']==0])}")

# Load new shoulder press data (CSV)
print("\n2. Loading new shoulder press data...")
csv_file = Path('../data/shoulder_press_features.csv')
df_shoulder = pd.read_csv(csv_file)
print(f"    Shoulder press samples: {len(df_shoulder)}")
print(f"      Correct: {len(df_shoulder[df_shoulder['label']==1])}")
print(f"      Incorrect: {len(df_shoulder[df_shoulder['label']==0])}")

# Combine datasets
print("\n3. Combining datasets...")
df_combined = pd.concat([df_bench[feature_columns + ['label']], 
                         df_shoulder[feature_columns + ['label']]], 
                        ignore_index=True)

print(f"\n    Combined dataset: {len(df_combined)} samples")
print(f"      Correct form: {len(df_combined[df_combined['label']==1])}")
print(f"      Incorrect form: {len(df_combined[df_combined['label']==0])}")

# Prepare features
X = df_combined[feature_columns].values
y = df_combined['label'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n4. Data split:")
print(f"   Training set: {len(X_train)} samples")
print(f"   Test set: {len(X_test)} samples")

# Normalize features
print("\n5. Normalizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest
print("\n6. Training Random Forest model...")
print("   (This may take 1-2 minutes...)")
rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_scaled, y_train)

# Evaluate
print("\n" + "="*60)
print("MODEL EVALUATION")
print("="*60)

train_pred = rf_model.predict(X_train_scaled)
train_accuracy = accuracy_score(y_train, train_pred)
print(f"\n Training Accuracy: {train_accuracy*100:.2f}%")

test_pred = rf_model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, test_pred)
print(f" Test Accuracy: {test_accuracy*100:.2f}%")

print("\nRunning 5-fold cross-validation...")
cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5)
print(f" CV Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*200:.2f}%)")

print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
print(classification_report(y_test, test_pred, target_names=['Incorrect', 'Correct']))

cm = confusion_matrix(y_test, test_pred)
print("\nConfusion Matrix:")
print(cm)
print(f"TN: {cm[0,0]} | FP: {cm[0,1]}")
print(f"FN: {cm[1,0]} | TP: {cm[1,1]}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n" + "="*60)
print("TOP 10 FEATURE IMPORTANCES")
print("="*60)
for idx, row in feature_importance.head(10).iterrows():
    print(f"{row['feature']:30s}: {row['importance']*100:5.2f}%")

# Save models
print("\n" + "="*60)
print("SAVING MODELS")
print("="*60)

models_dir = Path('../models')
models_dir.mkdir(exist_ok=True)

model_path = models_dir / 'combined_exercise_model.pkl'
scaler_path = models_dir / 'combined_exercise_scaler.pkl'

joblib.dump(rf_model, model_path)
joblib.dump(scaler, scaler_path)

print(f"\n Model saved: {model_path}")
print(f" Scaler saved: {scaler_path}")

# Save combined dataset
combined_csv_path = Path('../data/combined_training_data.csv')
df_combined.to_csv(combined_csv_path, index=False)
print(f" Combined dataset saved: {combined_csv_path}")

print("\n" + "="*60)
print(" TRAINING COMPLETE!")
print("="*60)
print(f"\nFinal Performance:")
print(f"  Training Accuracy: {train_accuracy*100:.2f}%")
print(f"  Test Accuracy: {test_accuracy*100:.2f}%")
print(f"  CV Accuracy: {cv_scores.mean()*100:.2f}%")
print(f"\n Model trained on:")
print(f"   • {len(df_bench)} bench press samples")
print(f"   • {len(df_shoulder)} shoulder press samples")
print(f"   • {len(df_combined)} total samples")
print(f"\n Ready to use in your real-time system!")
