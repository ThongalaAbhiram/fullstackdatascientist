import streamlit as st
import pandas as pd
import pickle
import numpy as np

model = pickle.load(open("artifacts/model.pkl","rb"))
scaler = pickle.load(open("artifacts/preprocessor.pkl","rb"))
metrics = pickle.load(open("artifacts/metrics.pkl","rb"))

st.title("💰 Loan Default Prediction")

st.subheader("Model Performance")

col1,col2,col3,col4 = st.columns(4)

col1.metric("Accuracy",round(metrics["accuracy"],2))
col2.metric("Precision",round(metrics["precision"],2))
col3.metric("Recall",round(metrics["recall"],2))
col4.metric("F1 Score",round(metrics["f1_score"],2))

st.divider()

cm = metrics["confusion_matrix"]

cm_df = pd.DataFrame(
cm,
index=["Actual No Default","Actual Default"],
columns=["Predicted No Default","Predicted Default"]
)

st.subheader("Confusion Matrix")

st.dataframe(cm_df)

st.divider()

st.subheader("Enter Customer Details")

income = st.number_input("Income")
loan_amount = st.number_input("Loan Amount")
credit_score = st.number_input("Credit Score")
age = st.number_input("Age")

if st.button("Predict"):

    data = np.array([[income,loan_amount,credit_score,age]])

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)

    if prediction[0]==1:
        st.error("High Risk of Default")

    else:
        st.success("Low Risk of Default")