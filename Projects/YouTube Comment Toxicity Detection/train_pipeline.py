import pandas as pd

from sklearn.model_selection import train_test_split

from src.data_ingestion import load_data
from src.preprocessing import clean_text

from src.feature_engineering import (
    vectorizer
)

from src.model_training import train_model

from src.evaluation import evaluate_model

from src.utils import save_artifacts

# --------------------------------------------
# LOAD DATA
# --------------------------------------------

df = load_data()

# --------------------------------------------
# CLEAN TEXT
# --------------------------------------------

df['comment_text'] = df[
    'comment_text'
].astype(str)

df['comment_text'] = df[
    'comment_text'
].apply(clean_text)

# --------------------------------------------
# FEATURES & LABELS
# --------------------------------------------

X = df['comment_text']

y = df['toxic']

# --------------------------------------------
# SPLIT
# --------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# --------------------------------------------
# TFIDF
# --------------------------------------------

X_train_vec = vectorizer.fit_transform(
    X_train
)

X_test_vec = vectorizer.transform(
    X_test
)

# --------------------------------------------
# TRAIN
# --------------------------------------------

model = train_model(
    X_train_vec,
    y_train
)

# --------------------------------------------
# EVALUATE
# --------------------------------------------

accuracy, report, matrix = evaluate_model(
    model,
    X_test_vec,
    y_test
)

print("Accuracy:", accuracy)

print(report)

# --------------------------------------------
# SAVE
# --------------------------------------------

save_artifacts(
    model,
    vectorizer
)