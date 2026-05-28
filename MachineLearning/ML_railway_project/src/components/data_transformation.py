import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from src.exception import CustomException

class DataTransformation:

    def initiate_data_transformation(self,raw_path):

        try:

            df = pd.read_csv(raw_path)

            X = df.drop("passenger_count",axis=1)

            y = df["passenger_count"]

            X_train,X_test,y_train,y_test = train_test_split(
                X,y,test_size=0.2,random_state=42
            )

            train = pd.concat([X_train,y_train],axis=1)

            test = pd.concat([X_test,y_test],axis=1)

            train.to_csv("artifacts/train.csv",index=False)

            test.to_csv("artifacts/test.csv",index=False)

            return "artifacts/train.csv","artifacts/test.csv"

        except Exception as e:

            raise CustomException(e,sys)