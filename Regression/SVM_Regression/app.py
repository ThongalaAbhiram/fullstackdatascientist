import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Page Config
st.set_page_config(
    page_title="SVM Regression",
    layout="centered"
)

# Load CSS
def load_css(file):
    with open(file) as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

load_css("style.css")

# Title
st.markdown("""
<div class="card">
<h1>SVM Regression</h1>

<p>
Predict <b>Tip Amount</b> from
<b>Total Bill</b> using
Support Vector Machine Regression (SVR)
</p>

</div>
""", unsafe_allow_html=True)

# Load Dataset
@st.cache_data
def load_data():
    return sns.load_dataset("tips")

df = load_data()

# Dataset Preview
st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("Dataset Preview")

st.dataframe(df.head())

st.markdown('</div>', unsafe_allow_html=True)

# Prepare Data
X = df[["total_bill"]]
y = df["tip"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Feature Scaling
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

y_train = scaler_y.fit_transform(
    y_train.values.reshape(-1, 1)
).ravel()

# Train Model
model = SVR(
    kernel="rbf",
    C=100,
    gamma=0.1,
    epsilon=0.1
)

model.fit(X_train, y_train)

# Predictions
y_pred_scaled = model.predict(X_test)

# Inverse Transform Predictions
y_pred = scaler_y.inverse_transform(
    y_pred_scaled.reshape(-1, 1)
).ravel()

# Metrics
mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)

adj_r2 = 1 - (
    (1 - r2) * (len(y_test) - 1)
    / (len(y_test) - X.shape[1] - 1)
)

# Visualization
st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("Total Bill vs Tip")

fig, ax = plt.subplots()

# Scatter Plot
ax.scatter(
    df["total_bill"],
    df["tip"],
    alpha=0.6
)

# Smooth Curve
X_grid = np.arange(
    min(df["total_bill"]),
    max(df["total_bill"]),
    0.1
)

X_grid = X_grid.reshape((len(X_grid), 1))

y_grid_scaled = model.predict(
    scaler_X.transform(X_grid)
)

y_grid = scaler_y.inverse_transform(
    y_grid_scaled.reshape(-1, 1)
)

ax.plot(
    X_grid,
    y_grid,
    color="red"
)

ax.set_xlabel("Total Bill")
ax.set_ylabel("Tip")

st.pyplot(fig)

st.markdown('</div>', unsafe_allow_html=True)

# Performance Metrics
st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("Model Performance")

c1, c2 = st.columns(2)

c1.metric("MAE", f"{mae:.2f}")
c2.metric("RMSE", f"{rmse:.2f}")

c3, c4 = st.columns(2)

c3.metric("R² Score", f"{r2:.2f}")
c4.metric("Adjusted R²", f"{adj_r2:.2f}")

st.markdown('</div>', unsafe_allow_html=True)

# Model Parameters
st.markdown(f"""
<div class="card">

<h3>SVR Parameters</h3>

<p><b>Kernel:</b> RBF</p>

<p><b>C:</b> 100</p>

<p><b>Gamma:</b> 0.1</p>

<p><b>Epsilon:</b> 0.1</p>

</div>
""", unsafe_allow_html=True)

# Prediction Section
st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("Predict Tip Amount")

bill = st.slider(
    "Total Bill ($)",
    float(df.total_bill.min()),
    float(df.total_bill.max()),
    30.0
)

# Convert Input
input_df = pd.DataFrame(
    [[bill]],
    columns=["total_bill"]
)

# Scale Input
scaled_input = scaler_X.transform(input_df)

# Predict
scaled_prediction = model.predict(scaled_input)

# Inverse Transform
prediction = scaler_y.inverse_transform(
    scaled_prediction.reshape(-1, 1)
)[0][0]

# Display Prediction
st.markdown(
    f'''
    <div class="prediction-box">
    Predicted Tip: ${prediction:.2f}
    </div>
    ''',
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)