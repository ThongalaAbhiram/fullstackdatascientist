import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="CNN CIFAR-10 Classifier",
    page_icon="🖼️"
)

# =========================
# CLASS LABELS
# =========================
classes = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck"
]

# =========================
# LOAD MODEL
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(
    BASE_DIR,
    "models",
    "cnn_cifar10_model.h5"
)

model = tf.keras.models.load_model(model_path)

# =========================
# TITLE
# =========================
st.title("🖼️ CNN CIFAR-10 Classifier")

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Resize
    image = image.resize((32,32))

    img_array = np.array(image)

    if len(img_array.shape) == 2:
        img_array = np.stack(
            (img_array,)*3,
            axis=-1
        )

    img_array = img_array / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    prediction = model.predict(img_array)

    predicted_class = np.argmax(
        prediction
    )

    st.success(
        f"Predicted Class: {classes[predicted_class]}"
    )