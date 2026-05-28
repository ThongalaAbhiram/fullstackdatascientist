import pandas as pd
from src.utils import load_object

class PredictionPipeline:

    def __init__(self):

        self.model = load_object("artifacts/model.pkl")

    def predict(self,data):

        df = pd.DataFrame([data])

        prediction = self.model.predict(df)

        return int(prediction[0])