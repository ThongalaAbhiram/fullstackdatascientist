import joblib

def save_model(model):

    joblib.dump(
        model,
        "artifacts/model.pkl"
    )