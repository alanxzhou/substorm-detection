import numpy as np
import matplotlib.pyplot as plt
import os.path
import utils
import rnn_models


plt.style.use('ggplot')

#my_path = os.path.abspath(os.path.dirname(__file__))
#path = os.path.join(my_path, "../../data/all_stations_data.npz")
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

X_train, X_val, X_test = utils.rnn_format_x([X_train, X_val, X_test])
y_train, y_val, y_test = utils.rnn_format_y([y_train,y_val,y_test])

params = {
    'epochs': 10,
    'batch_size': 32
}

hist, mod = rnn_models.train_basic_gru(X_train, y_train, X_val, y_val, params)