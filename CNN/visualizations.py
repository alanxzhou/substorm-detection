from CNN import models
import utils
import numpy as np
import keras
import matplotlib.pyplot as plt
import keras.backend as K

# CONFIGURATION
TRAIN = False

data_fn = "../data/2classes_data128_withsw_small.npz"
train_test_split = .11
train_val_split = .15
model_file = "saved models/model.h5"

# load in the data created by "create_dataset.py"
data = np.load(data_fn)
X = data['X']
y = data['y'][:, None]
strength = data['strength']
SW = data['SW']
ind = data['interval_index']

# create train, val and test sets
train, test = utils.split_data([X, SW, y, strength, ind], train_test_split, random=False)
del data
del X
del y
del strength
del SW
del ind
train, val = utils.split_data(train, train_val_split, random=True)
X_train, SW_train, y_train, strength_train, ind_train = train
X_val, SW_val, y_val, strength_val, ind_val = val
X_test, SW_test, y_test, strength_test, ind_test = test

train_data = [X_train, SW_train]
train_targets = [y_train, strength_train]
val_data = [X_val, SW_val]
val_targets = [y_val, strength_val]
idx = np.arange(X_test.shape[0])
np.random.shuffle(idx)
test_data = [X_test[idx], SW_test[idx]]
test_targets = [y_test[idx], strength_test[idx]]
ind_test = ind_test[idx]

print("X train shape:", X_train.shape, "proportion of substorms: ", np.mean(y_train))
print("X val shape:", X_val.shape, "proportion of substorms: ", np.mean(y_val))
print("X test shape:", X_test.shape, "proportion of substorms: ", np.mean(y_test))

params = {'batch_size': 16, 'epochs': 20, 'verbose': 2, 'n_classes': 2,
          'time_output_weight': 1000000, 'SW': True,

          'mag_T0': 128, 'mag_stages': 1, 'mag_blocks_per_stage': 4,
          'mag_downsampling_strides': (1, 2),
          'mag_kernel_size': (1, 11), 'mag_fl_filters': 48,
          'mag_fl_strides': (2, 2),
          'mag_fl_kernel_size': (2, 11), 'mag_type': 'residual',

          'sw_T0': 256, 'sw_stages': 3, 'sw_blocks_per_stage': 2,
          'sw_downsampling_strides': 2, 'sw_kernel_size': 11, 'sw_fl_filters': 48,
          'sw_fl_strides': 3, 'sw_fl_kernel_size': 11, 'sw_type': 'residual'}

if TRAIN:
    hist, mod = models.train_cnn(train_data, train_targets, val_data, val_targets, params)
    mod.summary()
    keras.models.save_model(mod, model_file)
    plt.subplot(211)
    plt.plot(hist.history['val_time_output_acc'])
    plt.subplot(212)
    plt.plot(hist.history['val_strength_output_mean_absolute_error'])
else:
    mod = keras.models.load_model(model_file,
                                  custom_objects={'true_positive': utils.true_positive,
                                                  'false_positive': utils.false_positive})
    mod.summary()

# Class activation maps
dense_weights = mod.get_layer('time_output').get_weights()[0][-96:, 0]
last_conv = K.function(mod.inputs, [mod.layers[-6].output])
n = 9
cam_data = [t[:n] for t in test_data]
cam_targets = test_targets[0][:n, 0]
predictions = mod.predict(cam_data)
cams = np.sum(last_conv(cam_data)[0] * dense_weights[None, None, None, :], axis=-1)
for i in range(3):
    for j in range(3):
        num = i * 3 + j
        plt.subplot(3, 3, num + 1)
        plt.pcolormesh(cams[num], vmin=cams.min(), vmax=cams.max())
        plt.title("P: {:4.2f}, L: {}, I: {}".format(predictions[0][num, 0], cam_targets[num], ind_test[num]))


# accuracy vs distance

# false positive

# false negative

# strength pred vs actual

plt.show()
