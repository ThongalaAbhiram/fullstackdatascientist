import streamlit as st
import pandas as pd
import plotly.express as px

from src.pipeline.prediction_pipeline import PredictionPipeline
from src.utils import recommend_coaches

# Page Config
st.set_page_config(
    page_title="Smart Railway Planner",
    page_icon="🚄",
    layout="wide"
)

# Title
st.title("🚄 Smart Railway Resource Planning System")
st.markdown("AI-powered system to **predict passenger demand** and optimize railway resources.")

# Load dataset
df = pd.read_csv("artifacts/raw.csv")

# --------------------------------------------------
# Sidebar Inputs
# --------------------------------------------------

st.sidebar.header("⚙ Train Details")

train_id = st.sidebar.number_input("Train ID", 0, 100, step=1)
route = st.sidebar.number_input("Route ID", 0, 10, step=1)
coaches = st.sidebar.slider("Number of Coaches", 8, 24)
platform = st.sidebar.slider("Platform Number", 1, 12)
weekend = st.sidebar.selectbox("Weekend", ["No", "Yes"])
delay = st.sidebar.slider("Delay Minutes", 0, 60)

weekend_val = 1 if weekend == "Yes" else 0

predict_btn = st.sidebar.button("🔮 Predict Demand")

# --------------------------------------------------
# KPI Metrics
# --------------------------------------------------

st.subheader("📊 Railway System Overview")

col1, col2, col3 = st.columns(3)

col1.metric("🚆 Total Trains", len(df))
col2.metric("👥 Avg Passengers", int(df["passenger_count"].mean()))
col3.metric("⏱ Avg Delay (min)", int(df["delay_minutes"].mean()))

st.divider()

# --------------------------------------------------
# Charts
# --------------------------------------------------

st.subheader("📈 Demand Analytics")

col1, col2 = st.columns(2)

route_demand = df.groupby("route")["passenger_count"].mean().reset_index()

fig_bar = px.bar(
    route_demand,
    x="route",
    y="passenger_count",
    color="passenger_count",
    title="Average Passenger Demand per Route",
    template="plotly_dark"
)

col1.plotly_chart(fig_bar, use_container_width=True)

fig_pie = px.pie(
    df,
    names="route",
    values="passenger_count",
    title="Passenger Distribution",
    hole=0.4
)

col2.plotly_chart(fig_pie, use_container_width=True)

# Line Chart
st.subheader("📊 Passenger Trends")

fig_line = px.line(
    df.sort_values("train_id"),
    x="train_id",
    y="passenger_count",
    markers=True,
    title="Passenger Trend by Train"
)

st.plotly_chart(fig_line, use_container_width=True)

st.divider()

# --------------------------------------------------
# Prediction Section
# --------------------------------------------------

st.subheader("🔮 AI Passenger Demand Prediction")

if predict_btn:

    data = {
        "train_id": train_id,
        "route": route,
        "num_coaches": coaches,
        "platform": platform,
        "weekend": weekend_val,
        "delay_minutes": delay
    }

    pipeline = PredictionPipeline()

    passengers = pipeline.predict(data)

    recommended = recommend_coaches(passengers)

    col1, col2 = st.columns(2)

    col1.metric("👥 Predicted Passengers", passengers)
    col2.metric("🚆 Recommended Coaches", recommended)

    if passengers > 900:
        st.error("⚠ High passenger demand expected! Consider adding more trains.")

    elif passengers < 300:
        st.success("Passenger demand is low.")

    else:
        st.info("Passenger demand is moderate.")