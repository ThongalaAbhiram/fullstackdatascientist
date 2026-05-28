import os
import joblib
import pandas as pd

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

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
    "knn_regressor.pkl"
)

scaler_path = os.path.join(
    model_dir,
    "scaler.pkl"
)

# =========================
# LOAD DATASET
# =========================
data = load_diabetes()

X = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

y = data.target

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
# MODEL
# =========================
model = KNeighborsRegressor(
    n_neighbors=5
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
# SCORE
# =========================
score = r2_score(y_test, y_pred)

print(f"R2 Score: {score:.2f}")

# =========================
# SAVE MODEL
# =========================
joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

print("✅ Model Saved Successfully!")