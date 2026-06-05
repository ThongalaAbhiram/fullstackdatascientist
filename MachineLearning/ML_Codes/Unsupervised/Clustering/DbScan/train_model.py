import os
import joblib
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestCentroid

# =========================
# BASE DIRECTORY
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================
# MODEL DIRECTORY
# =========================
model_dir = os.path.join(BASE_DIR, "models")
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(
    model_dir,
    "dbscan_model.pkl"
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
# FEATURE SCALING
# =========================
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# =========================
# DBSCAN MODEL
# =========================
dbscan = DBSCAN(
    eps=0.8,
    min_samples=5
)

cluster_labels = dbscan.fit_predict(
    X_scaled
)

# =========================
# REMOVE NOISE POINTS
# =========================
mask = cluster_labels != -1

X_filtered = X_scaled[mask]
labels_filtered = cluster_labels[mask]

# =========================
# CENTROID MODEL
# =========================
predictor = NearestCentroid()

predictor.fit(
    X_filtered,
    labels_filtered
)

# =========================
# SAVE MODEL
# =========================
joblib.dump(
    predictor,
    model_path
)

joblib.dump(
    scaler,
    scaler_path
)

print("✅ DBSCAN Model Saved Successfully!")    