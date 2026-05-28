import joblib

from src.preprocessing import clean_text

model = joblib.load(
    "artifacts/toxicity_model.pkl"
)

vectorizer = joblib.load(
    "artifacts/tfidf_vectorizer.pkl"
)

def predict_comment(comment):

    cleaned = clean_text(comment)

    vector = vectorizer.transform(
        [cleaned]
    )

    prediction = model.predict(vector)[0]

    probability = model.predict_proba(
        vector
    )[0]

    toxic_probability = round(
        probability[1] * 100,
        2
    )

    return prediction, toxic_probability