import streamlit as st
import numpy as np
import joblib
import os

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="KNN Regressor",
    page_icon="💉",
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
    "knn_regressor.pkl"
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
st.title("💉 KNN Regressor")
st.markdown("### Diabetes Progression Prediction")

st.write("---")

# =========================
# SIDEBAR
# =========================
st.sidebar.header("Enter Patient Details")

age = st.sidebar.slider(
    "Age",
    -0.2, 0.2, 0.0
)

sex = st.sidebar.slider(
    "Sex",
    -0.2, 0.2, 0.0
)

bmi = st.sidebar.slider(
    "BMI",
    -0.2, 0.2, 0.0
)

bp = st.sidebar.slider(
    "Blood Pressure",
    -0.2, 0.2, 0.0
)

s1 = st.sidebar.slider(
    "S1",
    -0.2, 0.2, 0.0
)

s2 = st.sidebar.slider(
    "S2",
    -0.2, 0.2, 0.0
)

s3 = st.sidebar.slider(
    "S3",
    -0.2, 0.2, 0.0
)

s4 = st.sidebar.slider(
    "S4",
    -0.2, 0.2, 0.0
)

s5 = st.sidebar.slider(
    "S5",
    -0.2, 0.2, 0.0
)

s6 = st.sidebar.slider(
    "S6",
    -0.2, 0.2, 0.0
)

# =========================
# INPUT DATA
# =========================
input_data = np.array([
    [
        age,
        sex,
        bmi,
        bp,
        s1,
        s2,
        s3,
        s4,
        s5,
        s6
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

    prediction = model.predict(input_scaled)

    st.success(
        f"Predicted Diabetes Progression Score: {prediction[0]:.2f}"
    )

st.write("---")
st.caption("Built with Streamlit & Scikit-learn")