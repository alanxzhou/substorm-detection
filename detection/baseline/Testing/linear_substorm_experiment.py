import numpy as np
import matplotlib.pyplot as plt
from detection import utils
import linear_models


plt.style.use('ggplot')

params = []

path = "D:\\substorm-detection\\data\\all_stations_data_128.npz"
data = np.load(path)
X = data['X']
y = data['y'][:, None]
train_test_split = .1
train_val_split = .15

# create train, val and test sets
train, test = utils.split_data([X, y], train_test_split, random=False, rnn_format=True)
train, val = utils.split_data(train, train_val_split, random=True, rnn_format=True)
X_train, y_train = train
X_val, y_val = val
X_test, y_test = test
X_train, X_val, X_test = utils.linear_format_x([X_train, X_val, X_test])
# train
score, history = linear_models.train_logistic_regression(X_train, y_train, X_val, y_val, params)
print(score)
print(history)