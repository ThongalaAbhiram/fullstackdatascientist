from sklearn.linear_model import LogisticRegression

def train_model(X_train, y_train):

    model = LogisticRegression(
        max_iter=5000,
        class_weight='balanced'
    )

    model.fit(X_train, y_train)

    return model