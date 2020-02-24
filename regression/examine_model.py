import keras
import numpy as np
from detection import utils
import pickle
import matplotlib.pyplot as plt

batch_size = 16
epochs = 40
mag_T0 = 96
sw_T0 = 196
params = {'batch_size': batch_size, 'epochs': epochs, 'verbose': 2,
          'time_output_weight': 1, 'SW': True,

          'Tm': mag_T0, 'mag_stages': 3, 'mag_blocks_per_stage': 3,
          'mag_downsampling_strides': (2, 2),
          'mag_kernel_size': (5, 5), 'mag_fl_filters': 16,
          'mag_fl_strides': (2, 2),
          'mag_fl_kernel_size': (5, 5), 'mag_type': 'basic',

          'Tw': sw_T0, 'sw_stages': 2, 'sw_blocks_per_stage': 2,
          'sw_downsampling_strides': 2, 'sw_kernel_size': 7, 'sw_fl_filters': 16,
          'sw_fl_strides': 3, 'sw_fl_kernel_size': 15, 'sw_type': 'residual'}

data_fn = "../data/regression_data13000.npz"
train_val_split = .15

data = np.load(data_fn)
X = data['mag_data_train'][:, :, :, 2:]
y = data['y_train'][:, None]
SW = data['sw_data_train']
X_test = data['mag_data_test'][:, :, :, 2:]
y_test = data['y_test'][:, None]
SW_test = data['sw_data_test']
train, val = utils.split_data([X, SW, y], train_val_split, random=True)
del data
del X
del y

X_train, SW_train, y_train = train
X_val, SW_val, y_val = val

train_data = [X_train, SW_train]
train_targets = y_train
val_data = [X_val, SW_val]
val_targets = y_val
test_data = [X_test[:, -params['Tm']:], SW_test[:, -params['Tw']:]]
test_targets = y_test

print("X train shape:", X_train.shape)
print("X val shape:", X_val.shape)
print("X test shape:", X_test.shape)

model = keras.models.load_model("regression_mod1.h5")
predictions = model.predict(test_data)
print()
