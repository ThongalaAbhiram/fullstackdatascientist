import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io

from sklearn.datasets import load_iris

import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Anomaly Detection Dashboard",
    page_icon="🚨",
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
    color:#D32F2F;
    text-align:center;
}

[data-testid="metric-container"]{
    background:#FFEBEE;
    border-radius:15px;
    padding:15px;
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

# =====================================
# LOAD MODEL
# =====================================

model = joblib.load(
    "models/isolation_forest.pkl"
)

scaler = joblib.load(
    "models/scaler.pkl"
)

# =====================================
# PREDICTIONS
# =====================================

X_scaled = scaler.transform(df)

predictions = model.predict(X_scaled)

df["Anomaly"] = predictions

df["Anomaly"] = df["Anomaly"].map(
    {
        1:"Normal",
        -1:"Anomaly"
    }
)

# =====================================
# TITLE
# =====================================

st.title("🚨 Anomaly Detection Dashboard")

st.markdown("---")

# =====================================
# METRICS
# =====================================

total_records = len(df)

anomalies = len(
    df[df["Anomaly"]=="Anomaly"]
)

normal = len(
    df[df["Anomaly"]=="Normal"]
)

anomaly_percent = (
    anomalies/total_records
)*100

col1,col2,col3,col4 = st.columns(4)

col1.metric(
    "Total Records",
    total_records
)

col2.metric(
    "Anomalies",
    anomalies
)

col3.metric(
    "Normal Records",
    normal
)

col4.metric(
    "Anomaly %",
    f"{anomaly_percent:.2f}%"
)

st.markdown("---")

# =====================================
# DATASET INFO
# =====================================

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

# =====================================
# VISUALIZATION 1
# =====================================

st.subheader(
    "Normal vs Anomaly"
)

fig1 = px.histogram(
    df,
    x="Anomaly",
    color="Anomaly"
)

st.plotly_chart(
    fig1,
    use_container_width=True
)

# =====================================
# VISUALIZATION 2
# =====================================

st.subheader(
    "Scatter Plot"
)

fig2 = px.scatter(
    df,
    x=df.columns[0],
    y=df.columns[2],
    color="Anomaly"
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

# =====================================
# VISUALIZATION 3
# =====================================

st.subheader(
    "Feature Distribution"
)

feature = st.selectbox(
    "Select Feature",
    df.columns[:-1]
)

fig3 = px.box(
    df,
    y=feature,
    color="Anomaly"
)

st.plotly_chart(
    fig3,
    use_container_width=True
)

# =====================================
# VISUALIZATION 4
# =====================================

st.subheader(
    "Correlation Heatmap"
)

fig4,ax = plt.subplots(
    figsize=(8,6)
)

sns.heatmap(
    df.drop(
        "Anomaly",
        axis=1
    ).corr(),
    annot=True,
    cmap="coolwarm",
    ax=ax
)

st.pyplot(fig4)

st.markdown("---")

st.success(
    "Isolation Forest successfully identified unusual observations within the dataset."
)