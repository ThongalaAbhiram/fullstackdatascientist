import streamlit as st
import numpy as np
import os
import joblib

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Linear Regression",
    page_icon="🏠",
    layout="centered"
)

# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(
    BASE_DIR,
    "models",
    "linear_regression.pkl"
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
st.title("🏠 Linear Regression")
st.subheader("California Housing Price Prediction")

st.write("---")

# =========================
# INPUTS
# =========================
MedInc = st.slider(
    "Median Income",
    0.0, 20.0, 5.0
)

HouseAge = st.slider(
    "House Age",
    1.0, 60.0, 20.0
)

AveRooms = st.slider(
    "Average Rooms",
    1.0, 15.0, 5.0
)

AveBedrms = st.slider(
    "Average Bedrooms",
    0.5, 5.0, 1.0
)

Population = st.slider(
    "Population",
    1, 40000, 1000
)

AveOccup = st.slider(
    "Average Occupancy",
    1.0, 10.0, 3.0
)

Latitude = st.slider(
    "Latitude",
    32.0, 42.0, 36.0
)

Longitude = st.slider(
    "Longitude",
    -125.0, -114.0, -120.0
)

# =========================
# INPUT DATA
# =========================
input_data = np.array([
    [
        MedInc,
        HouseAge,
        AveRooms,
        AveBedrms,
        Population,
        AveOccup,
        Latitude,
        Longitude
    ]
])

# =========================
# SCALE INPUT
# =========================
input_scaled = scaler.transform(
    input_data
)

# =========================
# PREDICTION
# =========================
if st.button("Predict House Price"):

    prediction = model.predict(
        input_scaled
    )

    st.success(
        f"Estimated House Price: ${prediction[0] * 100000:.2f}"
    )

st.write("---")
st.caption(
    "Built using Linear Regression"
)