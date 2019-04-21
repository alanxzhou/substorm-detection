import numpy as np
import utils_linear
from sklearn.linear_model import LogisticRegression


def train_logistic_regression(X_train, y_train, X_val, y_val, params):

    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    score = clf.score(X_val, y_val)

    y_val_pred = clf.predict(X_val)
    y_val_pred = np.reshape(y_val_pred, (len(y_val_pred),1))
    val_true_positive = utils_linear.true_positive(y_val, y_val_pred)
    val_false_positive = utils_linear.false_positive(y_val, y_val_pred)
    history = {
        'val_true_positive': val_true_positive,
        'val_false_positive': val_false_positive
    }

    return score, history
