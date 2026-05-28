import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load files
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
metrics = pickle.load(open("metrics.pkl", "rb"))

# Page config
st.set_page_config(page_title="Loan Default Predictor", page_icon="💰")

# Title
st.title("💰 Loan Default Prediction System")

st.write("Predict if a customer is likely to default on a loan.")

st.divider()

# ---------- MODEL METRICS ----------
st.subheader("📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", round(metrics["accuracy"], 2))
col2.metric("Precision", round(metrics["precision"], 2))
col3.metric("Recall", round(metrics["recall"], 2))
col4.metric("F1 Score", round(metrics["f1_score"], 2))

st.divider()

# Confusion Matrix
st.subheader("Confusion Matrix")

cm = metrics["confusion_matrix"]
cm_df = pd.DataFrame(
    cm,
    index=["Actual No Default", "Actual Default"],
    columns=["Predicted No Default", "Predicted Default"]
)

st.dataframe(cm_df)

st.divider()

# ---------- USER INPUT ----------
st.subheader("Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    income = st.number_input("Annual Income")
    credit_score = st.number_input("Credit Score")

with col2:
    loan_amount = st.number_input("Loan Amount")
    age = st.number_input("Age")

# Prediction
if st.button("Predict"):

    features = np.array([[income, loan_amount, credit_score, age]])
    scaled = scaler.transform(features)

    prediction = model.predict(scaled)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Loan Default")

    else:
        st.success("✅ Low Risk of Default")