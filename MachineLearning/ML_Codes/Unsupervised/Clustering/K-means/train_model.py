import os
import joblib
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# =========================
# BASE DIRECTORY
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_dir = os.path.join(BASE_DIR, "models")
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(
    model_dir,
    "kmeans_model.pkl"
)

scaler_path = os.path.join(
    model_dir,
    "scaler.pkl"
)

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
# KMEANS MODEL
# =========================
model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

model.fit(X_scaled)

# =========================
# SAVE MODEL
# =========================
joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

print("✅ KMeans Model Saved Successfully")