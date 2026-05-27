import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Page Config
st.set_page_config(page_title="Logistic Regression", layout="centered")

# Load CSS
def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# Title
st.markdown("""
<div class="card">
<h1>Logistic Regression</h1>
<p>Predict whether a passenger <b>Survived</b> or <b>Did Not Survive</b>
using Logistic Regression</p>
</div>
""", unsafe_allow_html=True)

# Load Dataset
@st.cache_data
def load_data():
    return sns.load_dataset("titanic")

df = load_data()

# Keep required columns only
df = df[["fare", "age", "survived"]]

# Remove missing values
df.dropna(inplace=True)

# Dataset Preview
st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.markdown('</div>', unsafe_allow_html=True)

# Prepare Data
X = df[["fare", "age"]]
y = df["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Visualization
st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("Fare vs Age")

fig, ax = plt.subplots()

scatter = ax.scatter(
    df["fare"],
    df["age"],
    c=df["survived"],
    alpha=0.7
)

ax.set_xlabel("Fare")
ax.set_ylabel("Age")

legend1 = ax.legend(
    *scatter.legend_elements(),
    title="Survived"
)

ax.add_artist(legend1)

st.pyplot(fig)

st.markdown('</div>', unsafe_allow_html=True)

# Performance
st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("Model Performance")

c1, c2 = st.columns(2)

c1.metric("Accuracy", f"{accuracy:.2f}")
c2.metric("Precision", f"{precision:.2f}")

c3, c4 = st.columns(2)

c3.metric("Recall", f"{recall:.2f}")
c4.metric("F1 Score", f"{f1:.2f}")

st.markdown('</div>', unsafe_allow_html=True)

# Confusion Matrix Display
st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("Confusion Matrix")

fig2, ax2 = plt.subplots()

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=ax2
)

ax2.set_xlabel("Predicted")
ax2.set_ylabel("Actual")

st.pyplot(fig2)

st.markdown('</div>', unsafe_allow_html=True)

# Model Coefficients
st.markdown(f"""
<div class="card">

<h3>Model Intercept & Coefficients</h3>

<p><b>Intercept:</b> {model.intercept_[0]:.3f}</p>

<p><b>Fare Coefficient:</b> {model.coef_[0][0]:.3f}</p>

<p><b>Age Coefficient:</b> {model.coef_[0][1]:.3f}</p>

</div>
""", unsafe_allow_html=True)

# Prediction Section
st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("Predict Survival")

fare = st.slider(
    "Passenger Fare ($)",
    float(df.fare.min()),
    float(df.fare.max()),
    50.0
)

age = st.slider(
    "Passenger Age",
    float(df.age.min()),
    float(df.age.max()),
    30.0
)

input_data = pd.DataFrame(
    [[fare, age]],
    columns=["fare", "age"]
)

prediction = model.predict(
    scaler.transform(input_data)
)[0]

probability = model.predict_proba(
    scaler.transform(input_data)
)[0][1]

# Output
if prediction == 1:
    st.markdown(
        f'<div class="prediction-box">✅ Passenger Survived '
        f'({probability*100:.2f}% probability)</div>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        f'<div class="prediction-box">❌ Passenger Did Not Survive '
        f'({(1-probability)*100:.2f}% probability)</div>',
        unsafe_allow_html=True
    )

st.markdown('</div>', unsafe_allow_html=True)