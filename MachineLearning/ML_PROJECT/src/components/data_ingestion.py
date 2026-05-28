import pandas as pd
import os
from sklearn.model_selection import train_test_split


class DataIngestion:

    def initiate_data_ingestion(self):

        df = pd.read_csv("notebook/data/dataset.csv")

        train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)

        os.makedirs("artifacts",exist_ok=True)

        train_set.to_csv("artifacts/train.csv",index=False)
        test_set.to_csv("artifacts/test.csv",index=False)

        return (
            "artifacts/train.csv",
            "artifacts/test.csv"
        )