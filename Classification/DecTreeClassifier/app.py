import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "models", "decision_tree_model.pkl")
scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Decision Tree Classifier",
    page_icon="🌳",
    layout="wide"
)

# ---------------- LOAD MODEL ---------------- #

model = joblib.load("models/decision_tree_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# ---------------- CUSTOM CSS ---------------- #

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

h1 {
    color: #00FFAA;
    text-align: center;
    font-size: 45px;
}

h3 {
    color: white;
}

.stButton>button {
    width: 100%;
    background-color: #00FFAA;
    color: black;
    border-radius: 10px;
    height: 3em;
    font-size: 18px;
    font-weight: bold;
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ---------------- #

st.title("🌳 Decision Tree Classifier")
st.write("### Heart Disease Prediction System")

st.write("---")

# ---------------- SIDEBAR ---------------- #

st.sidebar.header("Patient Information")

# ---------------- INPUTS ---------------- #

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 1, 100, 25)
    sex = st.selectbox("Sex", [0, 1])
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 600, 200)

with col2:
    fbs = st.selectbox("Fasting Blood Sugar", [0, 1])
    restecg = st.selectbox("Rest ECG", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate", 60, 250, 150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)

with col3:
    slope = st.selectbox("Slope", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thal", [0, 1, 2, 3])

# ---------------- PREDICTION ---------------- #

input_data = np.array([
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
]).reshape(1, -1)

if st.button("Predict Heart Disease"):

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    if prediction[0] == 1:

        st.markdown("""
        <div class='result-box' style='background-color:#ff4b4b;color:white;'>
        ⚠️ High Chance of Heart Disease
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div class='result-box' style='background-color:#00C853;color:white;'>
        ✅ Low Chance of Heart Disease
        </div>
        """, unsafe_allow_html=True)

# ---------------- FOOTER ---------------- #

st.write("---")
st.markdown(
    "<center>Built with ❤️ using Streamlit</center>",
    unsafe_allow_html=True
)