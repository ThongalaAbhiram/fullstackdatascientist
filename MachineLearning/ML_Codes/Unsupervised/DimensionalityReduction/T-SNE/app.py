import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="T-SNE Dashboard",
    page_icon="🧠",
    layout="wide"
)

# =====================================
# CSS
# =====================================

st.markdown("""
<style>

.main {
    background-color:#F4F8FB;
}

h1 {
    color:#1565C0;
    text-align:center;
}

[data-testid="metric-container"] {
    background-color:#E3F2FD;
    border-radius:15px;
    padding:15px;
}

</style>
""", unsafe_allow_html=True)

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
# SCALE DATA
# =====================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    df.drop("target", axis=1)
)

# =====================================
# TSNE
# =====================================

tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42
)

X_tsne = tsne.fit_transform(
    X_scaled
)

tsne_df = pd.DataFrame(
    X_tsne,
    columns=["TSNE1","TSNE2"]
)

tsne_df["target"] = iris.target

# =====================================
# TITLE
# =====================================

st.title("🧠 T-SNE Visualization Dashboard")

st.markdown("---")

# =====================================
# METRICS
# =====================================

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
    "Classes",
    len(np.unique(df["target"]))
)

col4.metric(
    "Embedding Dimensions",
    2
)

st.markdown("---")

# =====================================
# DATASET INFORMATION
# =====================================

tab1,tab2,tab3,tab4 = st.tabs([
    "Head",
    "Describe",
    "Info",
    "Shape"
])

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

# =====================================
# VISUALIZATION 1
# TSNE SCATTER
# =====================================

st.subheader("T-SNE Projection")

fig = px.scatter(
    tsne_df,
    x="TSNE1",
    y="TSNE2",
    color=tsne_df["target"].astype(str),
    title="2D T-SNE Visualization"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================
# VISUALIZATION 2
# CLASS DISTRIBUTION
# =====================================

st.subheader("Class Distribution")

fig2 = px.histogram(
    df,
    x="target"
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

# =====================================
# VISUALIZATION 3
# FEATURE DISTRIBUTION
# =====================================

st.subheader("Feature Distribution")

feature = st.selectbox(
    "Choose Feature",
    df.columns[:-1]
)

fig3 = px.histogram(
    df,
    x=feature,
    color=df["target"].astype(str)
)

st.plotly_chart(
    fig3,
    use_container_width=True
)

# =====================================
# VISUALIZATION 4
# HEATMAP
# =====================================

st.subheader("Correlation Heatmap")

fig4, ax = plt.subplots(
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
    "T-SNE transforms high-dimensional data into a lower-dimensional space while preserving local neighborhood structures."
)