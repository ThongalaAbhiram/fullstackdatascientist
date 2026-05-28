import joblib
import numpy as np

def predict_sales(data):

    model = joblib.load(
        "artifacts/model.pkl"
    )

    prediction = model.predict(
        np.array(data).reshape(1, -1)
    )

    return prediction[0]