from sklearn.linear_model import LogisticRegression


def train_logistic_regression(X_train, y_train, X_val, y_val, params):

    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    score = clf.score(X_val, y_val)

    return score
