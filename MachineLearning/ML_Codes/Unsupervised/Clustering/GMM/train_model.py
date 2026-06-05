import os
import joblib
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

# ====================================
# BASE DIRECTORY
# ====================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_dir = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(
    model_dir,
    exist_ok=True
)

# ====================================
# LOAD DATASET
# ====================================

iris = load_iris()

X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# ====================================
# SCALING
# ====================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ====================================
# GMM MODEL
# ====================================

gmm=GaussianMixture(
    n_components=3,
    covariance_type="full",
    random_state=42
)

gmm.fit(X_scaled)

# ====================================
# SAVE MODEL
# ====================================

joblib.dump(
    gmm,
    os.path.join(
        model_dir,
        "gmm_model.pkl"
    )
)

joblib.dump(
    scaler,
    os.path.join(
        model_dir,
        "scaler.pkl"
    )
)

print("✅ GMM Model Saved Successfully")