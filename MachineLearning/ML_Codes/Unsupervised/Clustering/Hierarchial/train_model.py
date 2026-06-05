import os
import joblib
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
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
    "hierarchical_model.pkl"
)

scaler_path = os.path.join(
    model_dir,
    "scaler.pkl"
)

# =========================
# LOAD IRIS DATASET
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
# HIERARCHICAL CLUSTERING
# =========================
cluster_model = AgglomerativeClustering(
    n_clusters=3,
    linkage="ward"
)

cluster_labels = cluster_model.fit_predict(
    X_scaled
)

# =========================
# CENTROID MODEL FOR PREDICTION
# =========================
predictor = NearestCentroid()

predictor.fit(
    X_scaled,
    cluster_labels
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

print("✅ Hierarchical Clustering Model Saved Successfully!")