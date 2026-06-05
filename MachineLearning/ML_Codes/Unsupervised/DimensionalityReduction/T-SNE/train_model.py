import os
import joblib
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

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
# T-SNE MODEL
# =========================
tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42
)

X_tsne = tsne.fit_transform(X_scaled)

# =========================
# SAVE
# =========================
joblib.dump(
    X_tsne,
    os.path.join(model_dir, "tsne_embedding.pkl")
)

joblib.dump(
    scaler,
    os.path.join(model_dir, "scaler.pkl")
)

print("✅ TSNE Embedding Saved Successfully!")