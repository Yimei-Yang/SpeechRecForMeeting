from sklearn.linear_model import LogisticRegression

def initialize(features, labels):
    features = np.stack(features, axis = 0)
    features = features.reshape(n, -1)
    return [LogisticRegression(), features, labels]

def train(model, x_train, y_train):
    fitted_model = model.fit(x_train, y_train)
    return fitted_model

def evaluate(model, x_test, y_test):
    score = logisticRegr.score(x_test, y_test)
    return [score]
