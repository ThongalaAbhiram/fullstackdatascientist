import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="RNN Sentiment Analysis",
    page_icon="🎬"
)

# =========================
# LOAD MODEL
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(
    BASE_DIR,
    "models",
    "rnn_model.h5"
)

model = tf.keras.models.load_model(
    model_path
)

# =========================
# WORD INDEX
# =========================
word_index = imdb.get_word_index()

# =========================
# TITLE
# =========================
st.title("🎬 RNN Sentiment Analysis")

st.write(
    "Enter a movie review:"
)

review = st.text_area(
    "Movie Review"
)

# =========================
# PREPROCESS
# =========================
def encode_review(text):

    words = text.lower().split()

    encoded = []

    for word in words:

        index = word_index.get(word)

        if index is not None and index < 10000:
            encoded.append(index + 3)

        else:
            encoded.append(2)   # unknown word

    return encoded
# =========================
# PREDICT
# =========================
if st.button("Predict Sentiment"):

    encoded_review = encode_review(
        review
    )

    padded_review = pad_sequences(
        [encoded_review],
        maxlen=200
    )

    prediction = model.predict(
        padded_review
    )[0][0]

    if prediction > 0.5:

        st.success(
            f"Positive Review ({prediction:.2f})"
        )

    else:

        st.error(
            f"Negative Review ({prediction:.2f})"
        )