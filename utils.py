import numpy as np
import keras.backend as K
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


def split_data(list_of_data, split, random=True, rnn_format=False):
    """this function splits a list of equal length (first dimension) data arrays into two lists. The length of the data
    put into the second list is determined by the 'split' argument. This can be used for slitting [X, y] into
    [X_train, y_train] and [X_val, y_val]
    """

    split_idx = int((1 - split) * list_of_data[0].shape[0])

    idx = np.arange(list_of_data[0].shape[0])
    if random:
        np.random.shuffle(idx)

    split_a = []
    split_b = []

    for data in list_of_data:
        split_a.append(data[idx[:split_idx]])
        split_b.append(data[idx[split_idx:]])
    return split_a, split_b


def true_positive(y_true, y_pred):
    y_pred_pos = K.round(y_pred[:, 0])
    y_pos = K.round(y_true[:, 0])
    return K.sum(y_pos * y_pred_pos) / (K.sum(y_pos) + K.epsilon())


def false_positive(y_true, y_pred):
    y_pred_pos = K.round(y_pred[:, 0])
    y_pos = K.round(y_true[:, 0])
    y_neg = 1 - y_pos
    return K.sum(y_pred_pos * y_neg) / (K.sum(y_neg) + K.epsilon())


def rnn_format_x(list_of_x):
    """
    reformats feature arrays to match input dimensions for RNNs
    :param list_of_x: list of feature arrays (e.g., X_train, X_val, etc.)
    :return: list of reshaped feature arrays
    """
    x_rnn = []
    for i in range(len(list_of_x)):
        a0, a1, a2, a3 = np.shape(list_of_x[i])
        x_rnn.append(np.reshape(list_of_x[i], (a0, a2, a1*a3)))
    return x_rnn


def rnn_format_y(list_of_y):
    """
    reformats labels into onehot encoding for rnn inputs
    :param list_of_y: list of label arrays (e.g., y_train, y_val, etc.)
    :return: y_rnn: list of label arrays in onehot encoding
    """
    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder(sparse=False)
    y_rnn = []
    for i in range(len(list_of_y)):
        integer_encoded = label_encoder.fit_transform(list_of_y[i])
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        y_rnn.append(onehot_encoder.fit_transform(integer_encoded))
    return y_rnn


def linear_format_x(list_of_x):
    """
    reformats list of feature arrays for linear classification
    :param list_of_x: list of feature arrays (e.g., X_train, X_val, etc.)
    :return: x_linear: list of feature arrays
    """
    x_linear = []
    for i in range(len(list_of_x)):
        a0, a1, a2, a3 = np.shape(list_of_x[i])
        x_linear.append(np.reshape(list_of_x[i], (a0, a1*a2*a3)))
    return x_linear