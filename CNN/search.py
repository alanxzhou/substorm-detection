import numpy as np
from CNN import models
import utils
import talos as ta

# CONFIGURATION
data_fn = "../data/all_stations_data_128.npz"
train_test_split = .11
train_val_split = .15

# load in the data created by "create_dataset.py"
data = np.load(data_fn)
X = data['X']
y = data['y'][:, None]

# create train, val and test sets
train, test = utils.split_data([X, y], train_test_split, random=False)
train, val = utils.split_data(train, train_val_split, random=True)
X_train, y_train = train
X_val, y_val = val
X_test, y_test = test

print("X train shape:", X_train.shape, "proportion of substorms: ", np.mean(y_train))
print("X val shape:", X_val.shape, "proportion of substorms: ", np.mean(y_val))
print("X test shape:", X_test.shape, "proportion of substorms: ", np.mean(y_test))
kernel_sizes = [(1, 5), (1, 7), (1, 9), (1, 11),
                (2, 5), (2, 7), (2, 9), (2, 11),
                (3, 5), (3, 7), (3, 9), (3, 11),
                (4, 5), (4, 7), (4, 9)]

fl_kernel_sizes = [(1, 13), (1, 7), (1, 9), (1, 11),
                   (2, 13), (2, 7), (2, 9), (2, 11),
                   (3, 13), (3, 7), (3, 9), (3, 11),
                   (4, 7), (4, 9), (4, 11), (4, 13)]

params = {'T0': [32, 64, 96, 128],
          'stages': [2, 3, 4, 5, 6],
          'blocks_per_stage': [1, 2, 3],
          'batch_size': [16, 32, 64],
          'epochs': [4, 8, 12, 16],
          'flx2': [True, False],
          'kernel_size': kernel_sizes,
          'downsampling_strides': [(1, 2), (1, 3), (2, 2), (2, 3)],
          'fl_filters': [16, 32, 64, 128],
          'fl_strides': [(1, 2), (1, 3), (2, 2), (2, 3)],
          'fl_kernel_size': fl_kernel_sizes,
          'verbose': [2]}

ta.Scan(X_train, y_train, params=params, model=models.train_strided_multistation_cnn,
        x_val=X_val, y_val=y_val, grid_downsample=4.52e-6)