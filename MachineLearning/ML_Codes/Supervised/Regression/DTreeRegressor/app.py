import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.tree import plot_tree

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Decision Tree Regressor",
    page_icon="🌳",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

h1 {
    color: #1f4e79;
    text-align: center;
    font-weight: bold;
}

h2, h3 {
    color: #0f172a;
}

[data-testid="stSidebar"] {
    background-color: #1e293b;
}

[data-testid="stSidebar"] * {
    color: white;
}

.stButton>button {
    width: 100%;
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    height: 3em;
    font-size: 18px;
    border: none;
}

.stButton>button:hover {
    background-color: #1d4ed8;
    color: white;
}

.metric-card {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.1);
    text-align: center;
}

.block-container {
    padding-top: 2rem;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# TITLE
# ==========================================

st.title("🌳 Decision Tree Regressor App")

st.markdown("""
<div style='text-align:center; color:gray; font-size:18px;'>
Predict House Prices using Machine Learning
</div>
""", unsafe_allow_html=True)

st.write("")

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("../data/kc_house_data.csv")

# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load(
    "models/decision_tree_regressor.pkl"
)

scaler = joblib.load(
    "models/scaler.pkl"
)

# ==========================================
# DATASET HEAD
# ==========================================

st.subheader("📊 Dataset Preview")

st.dataframe(
    df.head(),
    use_container_width=True
)

# ==========================================
# SIDEBAR INPUTS
# ==========================================

st.sidebar.header("🏠 Enter House Details")

bedrooms = st.sidebar.slider(
    "Bedrooms",
    1,
    10,
    3
)

bathrooms = st.sidebar.slider(
    "Bathrooms",
    1.0,
    10.0,
    2.0
)

sqft_living = st.sidebar.slider(
    "Sqft Living",
    500,
    10000,
    2000
)

floors = st.sidebar.slider(
    "Floors",
    1,
    4,
    1
)

waterfront = st.sidebar.selectbox(
    "Waterfront",
    [0,1]
)

view = st.sidebar.slider(
    "View",
    0,
    4,
    0
)

condition = st.sidebar.slider(
    "Condition",
    1,
    5,
    3
)

sqft_above = st.sidebar.slider(
    "Sqft Above",
    500,
    10000,
    1500
)

# ==========================================
# INPUT ARRAY
# ==========================================

input_data = np.array([[
    bedrooms,
    bathrooms,
    sqft_living,
    floors,
    waterfront,
    view,
    condition,
    sqft_above
]])

# ==========================================
# SCALE INPUT
# ==========================================

input_scaled = scaler.transform(
    input_data
)

# ==========================================
# PREDICT BUTTON
# ==========================================

if st.sidebar.button("Predict Price"):

    prediction = model.predict(
        input_scaled
    )

    st.subheader("🏡 Predicted House Price")

    st.markdown(f"""
    <div style="
        background-color:#dbeafe;
        padding:25px;
        border-radius:15px;
        text-align:center;
        font-size:35px;
        color:#1d4ed8;
        font-weight:bold;
    ">
        ${prediction[0]:,.2f}
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# MODEL METRICS
# ==========================================

X = df[[
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "floors",
    "waterfront",
    "view",
    "condition",
    "sqft_above"
]]

y = df["price"]

X_scaled = scaler.transform(X)

preds = model.predict(X_scaled)

mae = mean_absolute_error(y, preds)

mse = mean_squared_error(y, preds)

rmse = np.sqrt(mse)

r2 = r2_score(y, preds)

# ==========================================
# METRICS DISPLAY
# ==========================================

st.subheader("📈 Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "MAE",
        f"{mae:,.0f}"
    )

with col2:
    st.metric(
        "MSE",
        f"{mse:,.0f}"
    )

with col3:
    st.metric(
        "RMSE",
        f"{rmse:,.0f}"
    )

with col4:
    st.metric(
        "R² Score",
        f"{r2:.2f}"
    )

# ==========================================
# PRICE DISTRIBUTION
# ==========================================

st.subheader("📊 Price Distribution")

fig1, ax1 = plt.subplots(figsize=(10,5))

sns.histplot(
    df["price"],
    kde=True,
    ax=ax1
)

ax1.set_title("House Price Distribution")

st.pyplot(fig1)

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

st.subheader("⭐ Feature Importance")

importance = model.feature_importances_

feature_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_df = feature_df.sort_values(
    by="Importance",
    ascending=False
)

fig2, ax2 = plt.subplots(figsize=(10,5))

sns.barplot(
    x="Importance",
    y="Feature",
    data=feature_df,
    ax=ax2
)

ax2.set_title("Feature Importance")

st.pyplot(fig2)

# ==========================================
# DECISION TREE VISUALIZATION
# ==========================================

st.subheader("🌳 Decision Tree Visualization")

fig3, ax3 = plt.subplots(
    figsize=(25,12)
)

plot_tree(
    model,
    feature_names=X.columns,
    filled=True,
    rounded=True,
    fontsize=8,
    ax=ax3
)

st.pyplot(fig3)