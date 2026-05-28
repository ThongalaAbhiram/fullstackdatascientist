import streamlit as st

from src.prediction import predict_comment

st.set_page_config(
    page_title="YouTube Toxicity Detection",
    page_icon="💬"
)

st.title(
    "💬 YouTube Comment Toxicity Detection"
)

comment = st.text_area(
    "Enter Comment"
)

if st.button("Analyze"):

    prediction, toxic_probability = predict_comment(
        comment
    )

    if prediction == 1:

        st.error(
            f"Toxic Comment Detected ({toxic_probability}%)"
        )

    else:

        st.success(
            f"Non Toxic Comment ({100-toxic_probability}%)"
        )