import os
import tensorflow as tf

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_dir = os.path.join(BASE_DIR, "models")
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(
    model_dir,
    "rnn_model.h5"
)

# =========================
# LOAD DATASET
# =========================
vocab_size = 10000

(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=vocab_size
)

# =========================
# PAD SEQUENCES
# =========================
max_len = 200

x_train = pad_sequences(
    x_train,
    maxlen=max_len
)

x_test = pad_sequences(
    x_test,
    maxlen=max_len
)

# =========================
# MODEL
# =========================
model = Sequential([
    Embedding(
        input_dim=vocab_size,
        output_dim=32,
        input_length=max_len
    ),

    SimpleRNN(32),

    Dense(
        1,
        activation="sigmoid"
    )
])

# =========================
# COMPILE
# =========================
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# =========================
# TRAIN
# =========================
model.fit(
    x_train,
    y_train,
    epochs=3,
    batch_size=64,
    validation_data=(x_test, y_test)
)

# =========================
# SAVE
# =========================
model.save(model_path)

print("✅ Model Saved Successfully")