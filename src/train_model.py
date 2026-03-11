# model training utilities

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


def train_logistic_regression(X_train, y_train):

    # initialize model
    model = LogisticRegression(max_iter=1000)

    # train model
    model.fit(X_train, y_train)

    return model


def train_naive_bayes(X_train, y_train):

    # initialize model
    model = MultinomialNB()

    # train model
    model.fit(X_train, y_train)

    return model