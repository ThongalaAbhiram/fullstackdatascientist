import pandas as pd
import sys
from sklearn.ensemble import RandomForestRegressor
from src.exception import CustomException
from src.utils import save_object

class ModelTrainer:

    def initiate_model_trainer(self,train_path):

        try:

            train_df = pd.read_csv(train_path)

            X = train_df.drop("passenger_count",axis=1)

            y = train_df["passenger_count"]

            model = RandomForestRegressor()

            model.fit(X,y)

            save_object("artifacts/model.pkl",model)

            return model

        except Exception as e:

            raise CustomException(e,sys)