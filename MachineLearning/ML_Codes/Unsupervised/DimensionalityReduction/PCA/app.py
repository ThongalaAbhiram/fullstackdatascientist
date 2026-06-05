import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="PCA Dashboard",
    page_icon="📉",
    layout="wide"
)

# =====================================
# CSS
# =====================================

st.markdown("""
<style>

.main{
    background-color:#F5F7FA;
}

h1{
    color:#1565C0;
    text-align:center;
}

.metric-box{
    background:#E3F2FD;
    padding:20px;
    border-radius:15px;
}

</style>
""",
unsafe_allow_html=True)

# =====================================
# LOAD DATA
# =====================================

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df["target"] = iris.target

# =====================================
# LOAD MODEL
# =====================================

import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(
    os.path.join(
        BASE_DIR,
        "models",
        "pca_model.pkl"
    )
)

scaler = joblib.load(
    os.path.join(
        BASE_DIR,
        "models",
        "scaler.pkl"
    )
)


# =====================================
# PCA
# =====================================

X = df.drop(
    "target",
    axis=1
)

X_scaled = scaler.transform(X)

X_pca = pca.transform(X_scaled)

pca_df = pd.DataFrame(
    X_pca,
    columns=["PC1","PC2"]
)

pca_df["target"] = iris.target

# =====================================
# TITLE
# =====================================

st.title("📉 Principal Component Analysis Dashboard")

st.markdown("---")

# =====================================
# METRICS
# =====================================

variance = pca.explained_variance_ratio_

col1,col2,col3,col4 = st.columns(4)

col1.metric(
    "PC1 Variance",
    f"{variance[0]*100:.2f}%"
)

col2.metric(
    "PC2 Variance",
    f"{variance[1]*100:.2f}%"
)

col3.metric(
    "Total Variance",
    f"{sum(variance)*100:.2f}%"
)

col4.metric(
    "Original Features",
    X.shape[1]
)

st.markdown("---")

# =====================================
# DATASET INFO
# =====================================

tab1,tab2,tab3,tab4 = st.tabs([
    "Head",
    "Describe",
    "Info",
    "Shape"
])

with tab1:

    st.dataframe(
        df.head()
    )

with tab2:

    st.dataframe(
        df.describe()
    )

with tab3:

    buffer = io.StringIO()

    df.info(buf=buffer)

    st.text(
        buffer.getvalue()
    )

with tab4:

    st.write(
        f"Rows: {df.shape[0]}"
    )

    st.write(
        f"Columns: {df.shape[1]}"
    )

st.markdown("---")

# =====================================
# VISUALIZATION 1
# PCA SCATTER
# =====================================

st.subheader(
    "PCA Scatter Plot"
)

fig = px.scatter(
    pca_df,
    x="PC1",
    y="PC2",
    color=pca_df["target"].astype(str),
    title="PCA Projection"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================
# VISUALIZATION 2
# SCREE PLOT
# =====================================

st.subheader(
    "Scree Plot"
)

fig2 = px.bar(
    x=["PC1","PC2"],
    y=variance,
    labels={
        "x":"Principal Component",
        "y":"Variance Ratio"
    }
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

# =====================================
# VISUALIZATION 3
# CUMULATIVE VARIANCE
# =====================================

st.subheader(
    "Cumulative Variance"
)

cum_var = np.cumsum(
    variance
)

fig3 = px.line(
    x=[1,2],
    y=cum_var,
    markers=True
)

st.plotly_chart(
    fig3,
    use_container_width=True
)

# =====================================
# VISUALIZATION 4
# CORRELATION HEATMAP
# =====================================

st.subheader(
    "Correlation Heatmap"
)

fig4,ax = plt.subplots(
    figsize=(8,6)
)

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm",
    ax=ax
)

st.pyplot(fig4)

st.markdown("---")

st.success(
    "PCA reduced 4 features into 2 principal components while preserving maximum variance."
)