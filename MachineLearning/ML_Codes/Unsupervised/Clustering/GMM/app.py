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
    page_title="GMM Dashboard",
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
# LOAD DATA
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
        "gmm_model.pkl"
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

clusters = gmm.predict(X_scaled)

probabilities = gmm.predict_proba(X_scaled)

df["Cluster"] = clusters

# ====================================
# TITLE
# ====================================

st.title("🎯 Gaussian Mixture Model Dashboard")

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
    "Components",
    gmm.n_components
)

st.markdown("---")

# ====================================
# DATASET INFO
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

    st.text(buffer.getvalue())

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
    color=df["Cluster"].astype(str)
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

# ====================================
# VISUALIZATION 3
# PROBABILITY DISTRIBUTION
# ====================================

st.subheader(
    "Cluster Membership Probability"
)

prob_df = pd.DataFrame(
    probabilities,
    columns=[
        "Cluster 0",
        "Cluster 1",
        "Cluster 2"
    ]
)

st.bar_chart(
    prob_df.head(20)
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
    "GMM Parameters"
)

st.write(
    {
        "Number of Components":
        gmm.n_components,

        "Covariance Type":
        gmm.covariance_type,

        "Converged":
        gmm.converged_,

        "Iterations":
        gmm.n_iter_
    }
)

st.success(
    "GMM performs soft clustering by assigning probabilities to each cluster rather than hard assignments."
)