import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler


class DataTransformation:

    def initiate_data_transformation(self,train_path,test_path):

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        X_train = train_df.drop("default",axis=1)
        y_train = train_df["default"]

        X_test = test_df.drop("default",axis=1)
        y_test = test_df["default"]

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        pickle.dump(scaler,open("artifacts/preprocessor.pkl","wb"))

        train_arr = pd.concat(
            [pd.DataFrame(X_train_scaled),y_train.reset_index(drop=True)],axis=1
        ).values

        test_arr = pd.concat(
            [pd.DataFrame(X_test_scaled),y_test.reset_index(drop=True)],axis=1
        ).values

        return train_arr,test_arr