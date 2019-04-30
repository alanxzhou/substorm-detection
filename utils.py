import numpy as np
import keras.backend as K
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels


def split_data(list_of_data, split, random=True, batch_size=None):
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

    if batch_size:
        diff_a = len

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


def masked_mse(labels):
    def f(y_true, y_pred):
        return K.sum(K.square(y_true - y_pred)[labels])
    return f



def confusion_mtx(y_true, y_pred):
    y_true = np.ravel(np.array(y_true))
    y_pred = np.ravel(np.array(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    return cm


def plot_confusion_matrix(cm,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


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


def linear_format_y(list_of_y):
    """
    reformats list of label arrays for linear classification
    :param list_of_y: list of feature arrays (e.g., y_train, y_val, etc.)
    :return: x_linear: list of feature arrays
    """
    y_linear = []
    for i in range(len(list_of_y)):
        y_linear.append(np.ravel(list_of_y[i]))
    return y_linear

