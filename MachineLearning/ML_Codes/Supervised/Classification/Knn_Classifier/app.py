import streamlit as st
import numpy as np
import joblib
import os

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="KNN Classifier",
    page_icon="🩺",
    layout="centered"
)

# =========================
# BASE DIRECTORY
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================
# MODEL PATHS
# =========================
model_path = os.path.join(
    BASE_DIR,
    "models",
    "knn_classifier.pkl"
)

scaler_path = os.path.join(
    BASE_DIR,
    "models",
    "scaler.pkl"
)

# =========================
# LOAD MODEL
# =========================
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# =========================
# TITLE
# =========================
st.title("🩺 KNN Classifier")
st.markdown("### Breast Cancer Prediction System")

st.write("---")

# =========================
# SIDEBAR
# =========================
st.sidebar.header("Enter Patient Details")

mean_radius = st.sidebar.slider(
    "Mean Radius",
    5.0, 30.0, 14.0
)

mean_texture = st.sidebar.slider(
    "Mean Texture",
    5.0, 40.0, 20.0
)

mean_perimeter = st.sidebar.slider(
    "Mean Perimeter",
    40.0, 200.0, 90.0
)

mean_area = st.sidebar.slider(
    "Mean Area",
    100.0, 2500.0, 500.0
)

mean_smoothness = st.sidebar.slider(
    "Mean Smoothness",
    0.05, 0.20, 0.10
)

# =========================
# INPUT DATA
# =========================
input_data = np.array([
    [
        mean_radius,
        mean_texture,
        mean_perimeter,
        mean_area,
        mean_smoothness
    ]
])

# =========================
# DUMMY FEATURES
# =========================
# Dataset originally has 30 features.
# Fill remaining features with zeros.

remaining = np.zeros((1, 25))

input_data = np.concatenate(
    [input_data, remaining],
    axis=1
)

# =========================
# SCALING
# =========================
input_scaled = scaler.transform(input_data)

# =========================
# PREDICTION
# =========================
if st.button("Predict"):

    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.error("⚠️ Malignant Cancer Detected")
    else:
        st.success("✅ Benign Tumor Detected")

st.write("---")
st.caption("Built with Streamlit & Scikit-learn")