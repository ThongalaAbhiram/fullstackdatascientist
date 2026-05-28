import os
import pandas as pd
from src.logger import logging
from src.exception import CustomException
import sys

class DataIngestion:

    def __init__(self):

        self.dataset_path = "notebook/data/railway_data.csv"


    def initiate_data_ingestion(self):

        logging.info("Data ingestion started")

        try:

            df = pd.read_csv(self.dataset_path)

            os.makedirs("artifacts",exist_ok=True)

            df.to_csv("artifacts/raw.csv",index=False)

            logging.info("Data ingestion completed")

            return "artifacts/raw.csv"

        except Exception as e:

            raise CustomException(e,sys)