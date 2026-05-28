import pandas as pd
import joblib
from sklearn.metrics import r2_score


def save_object(file_path,obj):

    joblib.dump(obj,file_path)


def load_object(file_path):

    return joblib.load(file_path)


def evaluate_model(X_test,y_test,model):

    pred = model.predict(X_test)

    score = r2_score(y_test,pred)

    return score


def recommend_coaches(passengers):

    if passengers < 300:
        return 8
    elif passengers < 600:
        return 12
    elif passengers < 900:
        return 18
    else:
        return 24