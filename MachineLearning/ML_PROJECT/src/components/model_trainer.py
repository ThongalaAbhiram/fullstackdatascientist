import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix


class ModelTrainer:

    def initiate_model_trainer(self,train_arr,test_arr):

        X_train,y_train = train_arr[:,:-1],train_arr[:,-1]
        X_test,y_test = test_arr[:,:-1],test_arr[:,-1]

        model = RandomForestClassifier()

        model.fit(X_train,y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test,y_pred)
        precision = precision_score(y_test,y_pred,zero_division=0)
        recall = recall_score(y_test,y_pred,zero_division=0)
        f1 = f1_score(y_test,y_pred,zero_division=0)

        cm = confusion_matrix(y_test,y_pred,labels=[0,1])

        metrics = {
        "accuracy":accuracy,
        "precision":precision,
        "recall":recall,
        "f1_score":f1,
        "confusion_matrix":cm
        }

        pickle.dump(model,open("artifacts/model.pkl","wb"))
        pickle.dump(metrics,open("artifacts/metrics.pkl","wb"))

        return accuracy
    