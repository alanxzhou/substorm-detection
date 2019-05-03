from CNN import models
import utils
import numpy as np

########################################################################################################################
# CONFIGURATION
########################################################################################################################

data_fn = "../data/2classes_data128_withsw_small.npz"
train_test_split = .11
train_val_split = .15
model_file = "saved models/final_cnn_model.h5"

params = {'batch_size': 64, 'epochs': 15, 'verbose': 2, 'n_classes': 2,
          'time_output_weight': 1000000, 'SW': True,

          'mag_T0': 96, 'mag_stages': 1, 'mag_blocks_per_stage': 4,
          'mag_downsampling_strides': (2, 2),
          'mag_kernel_size': (2, 11), 'mag_fl_filters': 48,
          'mag_fl_strides': (3, 2),
          'mag_fl_kernel_size': (3, 11), 'mag_type': 'basic',

          'sw_T0': 128, 'sw_stages': 4, 'sw_blocks_per_stage': 3,
          'sw_downsampling_strides': 4, 'sw_kernel_size': 7, 'sw_fl_filters': 64,
          'sw_fl_strides': 4, 'sw_fl_kernel_size': 11, 'sw_type': 'residual'}

########################################################################################################################
# DATA LOADING
########################################################################################################################
data = np.load(data_fn)
X = data['X'][:, :, -params['mag_T0']:]
y = data['y'][:, None]
strength = data['strength']
SW = data['SW'][:, -params['sw_T0']:]
ind = data['interval_index']
st_loc = data['st_location'][:, :, -params['mag_T0']:]
ss_loc = data['ss_location']

# create train, val and test sets
train, test = utils.split_data([X, SW, y, strength, ind, st_loc, ss_loc], train_test_split, random=False)
del data
del X
del y
del strength
del SW
del ind
del st_loc
del ss_loc
train, val = utils.split_data(train, train_val_split, random=True)
X_train, SW_train, y_train, strength_train, ind_train, st_loc_train, ss_loc_train = train
X_val, SW_val, y_val, strength_val, ind_val, st_loc_val, ss_loc_val = val
X_test, SW_test, y_test, strength_test, ind_test, st_loc_test, ss_loc_test = test

train_data = [X_train, SW_train]
train_targets = [y_train, strength_train]
val_data = [X_val, SW_val]
val_targets = [y_val, strength_val]

idx = np.arange(X_test.shape[0])
np.random.shuffle(idx)
test_data = [X_test[idx], SW_test[idx]]
test_targets = [y_test[idx], strength_test[idx]]
ind_test = ind_test[idx]
st_loc_test = st_loc_test[idx]
ss_loc_test = ss_loc_test[idx]

print("X train shape:", X_train.shape, "proportion of substorms: ", np.mean(y_train))
print("X val shape:", X_val.shape, "proportion of substorms: ", np.mean(y_val))
print("X test shape:", X_test.shape, "proportion of substorms: ", np.mean(y_test))

########################################################################################################################
# MODEL
########################################################################################################################

N = 10
eval_length = 8
results = np.empty((N, eval_length))
for i in range(N):
    hist, mod = models.train_cnn(train_data, train_targets, val_data, val_targets, params)
    results[i] = mod.evaluate(test_data, test_targets)

print("Mean")
print(results.mean(axis=0))
print("Max")
print(results.max(axis=0))
print("Min")
print(results.min(axis=0))
print("STD")
print(results.std(axis=0))
