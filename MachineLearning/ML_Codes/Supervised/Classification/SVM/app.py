import streamlit as st
import numpy as np
import os
import joblib

# Page Config
st.set_page_config(
    page_title="SVM Classifier",
    page_icon="🩺",
    layout="centered"
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(
    BASE_DIR,
    "models",
    "svm_classifier.pkl"
)

scaler_path = os.path.join(
    BASE_DIR,
    "models",
    "scaler.pkl"
)

# Load Model
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Title
st.title("🩺 SVM Classifier")
st.subheader("Breast Cancer Prediction")

st.write("---")

# Inputs
radius = st.slider(
    "Mean Radius",
    5.0, 30.0, 14.0
)

texture = st.slider(
    "Mean Texture",
    5.0, 40.0, 20.0
)

perimeter = st.slider(
    "Mean Perimeter",
    40.0, 200.0, 90.0
)

area = st.slider(
    "Mean Area",
    100.0, 2500.0, 500.0
)

smoothness = st.slider(
    "Mean Smoothness",
    0.05, 0.20, 0.10
)

# Input Data
input_data = np.array([
    [
        radius,
        texture,
        perimeter,
        area,
        smoothness
    ]
])

# Remaining Features
remaining_features = np.zeros((1, 25))

input_data = np.concatenate(
    [input_data, remaining_features],
    axis=1
)

# Scaling
input_scaled = scaler.transform(
    input_data
)

# Prediction
if st.button("Predict"):

    prediction = model.predict(
        input_scaled
    )[0]

    probability = model.predict_proba(
        input_scaled
    )[0]

    if prediction == 1:
        st.success("✅ Benign Tumor")
    else:
        st.error("⚠️ Malignant Tumor")

    st.write(
        f"Confidence: {max(probability)*100:.2f}%"
    )

st.write("---")
st.caption("Built using SVM Classification")