# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

from sklearn.metrics import accuracy_score

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Random Forest Classifier",
    page_icon="🌲",
    layout="centered"
)

# =========================
# PATHS
# =========================
BASE_DIR = Path(__file__).parent

model_path = BASE_DIR / "models" / "random_forest_model.pkl"
scaler_path = BASE_DIR / "models" / "scaler.pkl"
data_path = BASE_DIR / "data" / "heart.csv"

# =========================
# LOAD MODEL & SCALER
# =========================
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(data_path)

# =========================
# TITLE
# =========================
st.title("🌲 Random Forest Classifier")
st.markdown("### Heart Disease Prediction System")

st.write("---")

# =========================
# SIDEBAR
# =========================
st.sidebar.header("Enter Patient Details")

# =========================
# INPUT FIELDS
# =========================
age = st.sidebar.slider("Age", 20, 100, 45)

sex = st.sidebar.selectbox(
    "Sex",
    ["Female", "Male"]
)
sex = 1 if sex == "Male" else 0

cp = st.sidebar.selectbox(
    "Chest Pain Type",
    [0, 1, 2, 3]
)

trestbps = st.sidebar.slider(
    "Resting Blood Pressure",
    80, 200, 120
)

chol = st.sidebar.slider(
    "Cholesterol",
    100, 600, 200
)

fbs = st.sidebar.selectbox(
    "Fasting Blood Sugar > 120",
    [0, 1]
)

restecg = st.sidebar.selectbox(
    "Rest ECG",
    [0, 1, 2]
)

thalach = st.sidebar.slider(
    "Maximum Heart Rate",
    60, 220, 150
)

exang = st.sidebar.selectbox(
    "Exercise Induced Angina",
    [0, 1]
)

oldpeak = st.sidebar.slider(
    "Oldpeak",
    0.0, 6.0, 1.0
)

slope = st.sidebar.selectbox(
    "Slope",
    [0, 1, 2]
)

ca = st.sidebar.selectbox(
    "Number of Major Vessels",
    [0, 1, 2, 3, 4]
)

thal = st.sidebar.selectbox(
    "Thalassemia",
    [0, 1, 2, 3]
)

# =========================
# CREATE INPUT ARRAY
# =========================
input_data = np.array([
    [
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]
])

# =========================
# SCALING
# =========================
input_scaled = scaler.transform(input_data)

# =========================
# PREDICTION
# =========================
if st.button("Predict"):

    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)

    st.write("---")

    if prediction == 1:
        st.error("⚠️ High Chances of Heart Disease")
    else:
        st.success("✅ Low Chances of Heart Disease")

    st.subheader("Prediction Probability")

    st.write(
        f"Probability of Heart Disease: "
        f"{prediction_proba[0][1]*100:.2f}%"
    )

# =========================
# DATA PREVIEW
# =========================
with st.expander("View Dataset"):
    st.dataframe(df.head())

# =========================
# FOOTER
# =========================
st.write("---")
st.caption("Built with Streamlit & Scikit-learn")