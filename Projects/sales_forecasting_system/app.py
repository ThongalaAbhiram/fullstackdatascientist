import streamlit as st
import pandas as pd

from src.prediction import predict_sales

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="Sales Forecasting System",
    page_icon="📈",
    layout="wide"
)

# ------------------------------------------------
# CSS
# ------------------------------------------------

st.markdown("""
<style>

.stApp{
    background-color:#f8fafc;
}

.main-title{
    text-align:center;
    font-size:42px;
    font-weight:bold;
    color:#1e293b;
}

.sub-title{
    text-align:center;
    color:#64748b;
    margin-bottom:40px;
}

.stButton>button{
    width:100%;
    height:50px;
    background-color:#2563eb;
    color:white;
    border:none;
    border-radius:10px;
    font-size:18px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# TITLE
# ------------------------------------------------

st.markdown(
    '<div class="main-title">📈 Sales Forecasting System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Predict Future Sales using Machine Learning</div>',
    unsafe_allow_html=True
)

# ------------------------------------------------
# INPUTS
# ------------------------------------------------

month = st.selectbox(
    "Select Month",
    [
        "January","February","March",
        "April","May","June",
        "July","August","September",
        "October","November","December"
    ]
)

region = st.selectbox(
    "Select Region",
    ["North","South","East","West"]
)

product = st.selectbox(
    "Select Product",
    ["Laptop","Mobile","Tablet"]
)

advertising = st.number_input(
    "Advertising Spend",
    min_value=1000
)

units = st.number_input(
    "Units Sold",
    min_value=1
)

# ------------------------------------------------
# MANUAL ENCODING
# ------------------------------------------------

month_map = {
    "January":0,"February":1,"March":2,
    "April":3,"May":4,"June":5,
    "July":6,"August":7,"September":8,
    "October":9,"November":10,"December":11
}

region_map = {
    "North":0,
    "South":1,
    "East":2,
    "West":3
}

product_map = {
    "Laptop":0,
    "Mobile":1,
    "Tablet":2
}

# ------------------------------------------------
# PREDICTION
# ------------------------------------------------

if st.button("Predict Sales"):

    input_data = [
        month_map[month],
        region_map[region],
        product_map[product],
        advertising,
        units
    ]

    prediction = predict_sales(
        input_data
    )

    st.success(
        f"Predicted Sales: ₹ {round(prediction,2)}"
    )