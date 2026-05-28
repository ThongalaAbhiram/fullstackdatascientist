import os
import joblib
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
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
    "decision_tree_regressor.pkl"
)

scaler_path = os.path.join(
    model_dir,
    "scaler.pkl"
)

# =========================
# LOAD DATASET
# =========================
housing = fetch_california_housing()

X = pd.DataFrame(
    housing.data,
    columns=housing.feature_names
)

y = housing.target

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
model = DecisionTreeRegressor(
    max_depth=8,
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