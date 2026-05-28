import joblib

def save_artifacts(
    model,
    vectorizer
):

    joblib.dump(
        model,
        "artifacts/toxicity_model.pkl"
    )

    joblib.dump(
        vectorizer,
        "artifacts/tfidf_vectorizer.pkl"
    )