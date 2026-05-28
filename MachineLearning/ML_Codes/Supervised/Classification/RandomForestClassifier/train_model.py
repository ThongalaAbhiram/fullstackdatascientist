# train_model.py

import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =========================
# BASE DIRECTORY
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================
# FILE PATHS
# =========================
data_path = os.path.join(BASE_DIR, "data", "heart.csv")

model_dir = os.path.join(BASE_DIR, "models")

model_path = os.path.join(
    model_dir,
    "random_forest_model.pkl"
)

scaler_path = os.path.join(
    model_dir,
    "scaler.pkl"
)

# =========================
# CREATE MODELS DIRECTORY
# =========================
os.makedirs(model_dir, exist_ok=True)

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv(data_path)

# =========================
# FEATURES & TARGET
# =========================
X = df.drop("target", axis=1)
y = df["target"]

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# FEATURE SCALING
# =========================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# RANDOM FOREST MODEL
# =========================
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# =========================
# TRAIN MODEL
# =========================
model.fit(X_train_scaled, y_train)

# =========================
# PREDICTIONS
# =========================
y_pred = model.predict(X_test_scaled)

# =========================
# ACCURACY
# =========================
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2f}")

# =========================
# SAVE MODEL & SCALER
# =========================
joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

print("✅ Model saved successfully!")
print("✅ Scaler saved successfully!")