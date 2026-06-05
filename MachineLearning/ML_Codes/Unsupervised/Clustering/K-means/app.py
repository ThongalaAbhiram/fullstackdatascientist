import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io

from sklearn.datasets import load_iris

import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# ====================================
# PAGE CONFIG
# ====================================

st.set_page_config(
    page_title="K-Means Dashboard",
    page_icon="🎯",
    layout="wide"
)

# ====================================
# CSS
# ====================================

st.markdown("""
<style>

.main{
    background-color:#F5F7FA;
}

h1{
    color:#1565C0;
    text-align:center;
}

[data-testid="metric-container"]{
    background:#E3F2FD;
    border-radius:15px;
    padding:15px;
}

</style>
""",
unsafe_allow_html=True)

# ====================================
# LOAD DATASET
# ====================================

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# ====================================
# LOAD MODEL
# ====================================

import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(
    os.path.join(
        BASE_DIR,
        "models",
        "kmeans_model.pkl"
    )
)

scaler = joblib.load(
    os.path.join(
        BASE_DIR,
        "models",
        "scaler.pkl"
    )
)
# ====================================
# CLUSTERING
# ====================================

X_scaled = scaler.transform(df)

clusters = model.predict(X_scaled)

df["Cluster"] = clusters

# ====================================
# TITLE
# ====================================

st.title("🎯 K-Means Clustering Dashboard")

st.markdown("---")

# ====================================
# METRICS
# ====================================

col1,col2,col3,col4 = st.columns(4)

col1.metric(
    "Rows",
    df.shape[0]
)

col2.metric(
    "Features",
    df.shape[1]-1
)

col3.metric(
    "Clusters",
    len(np.unique(clusters))
)

col4.metric(
    "Inertia",
    f"{model.inertia_:.2f}"
)

st.markdown("---")

# ====================================
# DATASET INFORMATION
# ====================================

tab1,tab2,tab3,tab4 = st.tabs(
    [
        "Head",
        "Describe",
        "Info",
        "Shape"
    ]
)

with tab1:
    st.dataframe(df.head())

with tab2:
    st.dataframe(df.describe())

with tab3:

    buffer = io.StringIO()

    df.info(buf=buffer)

    st.text(
        buffer.getvalue()
    )

with tab4:

    st.write(
        f"Rows : {df.shape[0]}"
    )

    st.write(
        f"Columns : {df.shape[1]}"
    )

st.markdown("---")

# ====================================
# VISUALIZATION 1
# CLUSTER DISTRIBUTION
# ====================================

st.subheader(
    "Cluster Distribution"
)

fig1 = px.histogram(
    df,
    x="Cluster",
    color="Cluster"
)

st.plotly_chart(
    fig1,
    use_container_width=True
)

# ====================================
# VISUALIZATION 2
# SCATTER PLOT
# ====================================

st.subheader(
    "Cluster Visualization"
)

fig2 = px.scatter(
    df,
    x=df.columns[0],
    y=df.columns[2],
    color=df["Cluster"].astype(str),
    title="K-Means Clusters"
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

# ====================================
# VISUALIZATION 3
# PAIRWISE FEATURE VIEW
# ====================================

st.subheader(
    "Feature Relationship"
)

feature_x = st.selectbox(
    "X Axis",
    iris.feature_names,
    index=0
)

feature_y = st.selectbox(
    "Y Axis",
    iris.feature_names,
    index=2
)

fig3 = px.scatter(
    df,
    x=feature_x,
    y=feature_y,
    color=df["Cluster"].astype(str)
)

st.plotly_chart(
    fig3,
    use_container_width=True
)

# ====================================
# VISUALIZATION 4
# HEATMAP
# ====================================

st.subheader(
    "Correlation Heatmap"
)

fig4,ax = plt.subplots(
    figsize=(8,6)
)

sns.heatmap(
    df.drop(
        "Cluster",
        axis=1
    ).corr(),
    annot=True,
    cmap="coolwarm",
    ax=ax
)

st.pyplot(fig4)

st.markdown("---")

# ====================================
# MODEL INFORMATION
# ====================================

st.subheader(
    "K-Means Model Information"
)

st.write(
    {
        "Clusters": model.n_clusters,
        "Iterations": model.n_iter_,
        "Inertia": round(model.inertia_,2)
    }
)

# ====================================
# NEW SAMPLE PREDICTION
# ====================================

st.markdown("---")

st.subheader(
    "Predict Cluster for New Flower"
)

col1,col2 = st.columns(2)

with col1:

    sepal_length = st.slider(
        "Sepal Length",
        4.0,8.0,5.5
    )

    sepal_width = st.slider(
        "Sepal Width",
        2.0,5.0,3.0
    )

with col2:

    petal_length = st.slider(
        "Petal Length",
        1.0,7.0,3.5
    )

    petal_width = st.slider(
        "Petal Width",
        0.1,3.0,1.0
    )

if st.button(
    "Predict Cluster",
    use_container_width=True
):

    sample = np.array([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])

    sample_scaled = scaler.transform(
        sample
    )

    prediction = model.predict(
        sample_scaled
    )[0]

    st.success(
        f"Predicted Cluster : {prediction}"
    )