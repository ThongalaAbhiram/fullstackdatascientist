import pandas as pd

def load_data():

    df = pd.read_csv(
        "dataset/train.csv"
    )

    return df