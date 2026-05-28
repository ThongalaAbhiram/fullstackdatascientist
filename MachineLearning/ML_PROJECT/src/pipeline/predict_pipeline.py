import pickle
import numpy as np


class PredictPipeline:

    def predict(self,features):

        model = pickle.load(open("artifacts/model.pkl","rb"))
        scaler = pickle.load(open("artifacts/preprocessor.pkl","rb"))

        data_scaled = scaler.transform(features)

        pred = model.predict(data_scaled)

        return pred