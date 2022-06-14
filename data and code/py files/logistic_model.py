import torch
from sklearn.linear_model import roc_curve

class LogisticRegression(torch.nn.Module):
     def __init__(self, input_dim, output_dim):
         super(LogisticRegression, self).__init__()
         self.linear = torch.nn.Linear(input_dim, output_dim)     def forward(self, x):
         outputs = torch.sigmoid(self.linear(x))
         return outputs



def initialize(features):quit()
    n = features.size(0)
    p = features.size(1)*features.size(2)
    features = torch.reshape(features, (n, p))
    return [LogisticRegression(), features, labels]

def train(model, x_train, y_train):
    fitted_model = model.fit(x_train, y_train)
    return fitted_model

def evaluate(model, x_test, y_test, y_hat):
    score = logisticRegr.score(x_test, y_test)
    y_hat = model.predict(y_test)
    [fpr, tpr, thresholds] = roc_curve(labels, y_hat)
    return [score, y_hat, fpr, tpr, thresholds]
