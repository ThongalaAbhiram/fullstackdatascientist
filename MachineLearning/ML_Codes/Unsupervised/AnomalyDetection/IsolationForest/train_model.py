import os
import joblib
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# =========================
# BASE DIRECTORY
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_dir = os.path.join(BASE_DIR, "models")
os.makedirs(model_dir, exist_ok=True)

# =========================
# LOAD DATASET
# =========================
iris = load_iris()

X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# =========================
# SCALING
# =========================
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# =========================
# MODEL
# =========================
model = IsolationForest(
    contamination=0.05,
    random_state=42
)

model.fit(X_scaled)

# =========================
# SAVE
# =========================
joblib.dump(
    model,
    os.path.join(model_dir, "isolation_forest.pkl")
)

joblib.dump(
    scaler,
    os.path.join(model_dir, "scaler.pkl")
)

print("✅ Isolation Forest Model Saved")