import numpy as np


def true_positive(y_true, y_pred):
    y_pos = y_true
    y_pred_pos = y_pred
    return np.sum(y_pos * y_pred_pos) / (np.sum(y_pos) + 1e-10)


def false_positive(y_true, y_pred):
    y_pos = y_true
    y_pred_pos = y_pred
    y_neg = 1 - y_pos
    return np.sum(y_pred_pos * y_neg) / (np.sum(y_neg) + 1e-10)


class History:
    def __init__(self, history_dict):
        self.history = history_dict
