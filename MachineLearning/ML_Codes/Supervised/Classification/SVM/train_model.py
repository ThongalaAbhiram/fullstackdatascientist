import os
import joblib
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Base Directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Models Folder
model_dir = os.path.join(BASE_DIR, "models")
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "svm_classifier.pkl")
scaler_path = os.path.join(model_dir, "scaler.pkl")

# Load Dataset
data = load_breast_cancer()

X = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

y = data.target

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVM Model
model = SVC(
    kernel="rbf",
    probability=True,
    random_state=42
)

# Train
model.fit(
    X_train_scaled,
    y_train
)

# Evaluation
y_pred = model.predict(
    X_test_scaled
)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(f"Accuracy: {accuracy:.4f}")

# Save
joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

print("✅ Model Saved Successfully")