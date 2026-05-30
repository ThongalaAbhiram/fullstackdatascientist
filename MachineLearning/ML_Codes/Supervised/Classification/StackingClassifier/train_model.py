import os
import joblib
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score

# =========================
# BASE DIRECTORY
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_dir = os.path.join(BASE_DIR, "models")
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(
    model_dir,
    "stacking_classifier.pkl"
)

scaler_path = os.path.join(
    model_dir,
    "scaler.pkl"
)

# =========================
# LOAD DATASET
# =========================
data = load_breast_cancer()

X = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

y = data.target

# =========================
# SPLIT DATA
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# SCALING
# =========================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# BASE LEARNERS
# =========================
base_models = [
    ("dt", DecisionTreeClassifier(max_depth=5)),
    ("knn", KNeighborsClassifier(n_neighbors=5)),
    ("lr", LogisticRegression(max_iter=1000))
]

# =========================
# META LEARNER
# =========================
meta_model = LogisticRegression()

# =========================
# STACKING MODEL
# =========================
model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model
)

# =========================
# TRAIN
# =========================
model.fit(X_train_scaled, y_train)

# =========================
# EVALUATE
# =========================
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(f"Accuracy: {accuracy:.4f}")

# =========================
# SAVE MODEL
# =========================
joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

print("✅ Model Saved Successfully")