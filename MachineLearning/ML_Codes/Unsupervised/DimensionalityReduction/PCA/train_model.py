import os
import joblib
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_dir = os.path.join(BASE_DIR, "models")

os.makedirs(model_dir, exist_ok=True)

iris = load_iris()

X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)

pca.fit(X_scaled)

joblib.dump(
    pca,
    os.path.join(model_dir, "pca_model.pkl")
)

joblib.dump(
    scaler,
    os.path.join(model_dir, "scaler.pkl")
)

print("PCA Model Saved Successfully!")