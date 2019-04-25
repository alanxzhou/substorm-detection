import numpy as np
from utils import confusion_mtx
from sklearn.linear_model import LogisticRegression


def train_logistic_regression(X_train, y_train, X_val, y_val, params):
    """
    Trains logistic regression model
    :param X_train:
    :param y_train:
    :param X_val:
    :param y_val:
    :param params:
    :return:
        score: accuracy on validation set
        cm: numpy array, shape = (2,2)
            confusion matrix
        mod: trained model
    """
    clf = LogisticRegression(C=params['C'], solver='lbfgs')
    clf.fit(X_train, y_train)
    score = clf.score(X_val, y_val)

    y_val_pred = clf.predict(X_val)
    y_val_pred = np.ravel(y_val_pred)

    cm = confusion_mtx(y_val, y_val_pred)

    return score, cm, clf
