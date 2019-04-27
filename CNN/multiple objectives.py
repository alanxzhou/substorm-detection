import numpy as np
import matplotlib.pyplot as plt
from CNN import models
import utils
plt.style.use('ggplot')


# CONFIGURATION
data_fn = "../data/1classes_data64_withsw.npz"
batch_size = 16
epochs = 10
mag_T0 = 64
sw_T0 = 240
train_test_split = .11
train_val_split = .15

# load in the data created by "create_dataset.py"
data = np.load(data_fn)
X = data['X']
y = data['y'][:, None]
strength = data['strength']
SW = data['SW']

# create train, val and test sets
train, test = utils.split_data([X, SW, y, strength], train_test_split, random=False)
train, val = utils.split_data(train, train_val_split, random=True)
X_train, SW_train, y_train, strength_train = train
X_val, SW_val, y_val, strength_val = val
X_test, SW_test, y_test, strength_test = test

print("X train shape:", X_train.shape, "proportion of substorms: ", np.mean(y_train))
print("X val shape:", X_val.shape, "proportion of substorms: ", np.mean(y_val))
print("X test shape:", X_test.shape, "proportion of substorms: ", np.mean(y_test))

params = {'batch_size': batch_size, 'epochs': epochs, 'verbose': 2, 'n_classes': 2,
          'mag_T0': mag_T0, 'mag_stages': 2, 'mag_blocks_per_stage': 4, 'mag_downsampling_strides': [2, 2],
          'mag_kernel_size': [3, 5], 'mag_fl_filters': 64, 'mag_fl_strides': [1, 1],
          'mag_fl_kernel_size': [2, 13],

          'sw_T0': sw_T0, 'sw_stages': 2, 'sw_blocks_per_stage': 4, 'sw_downsampling_strides': 2,
          'sw_kernel_size': 7, 'sw_fl_filters': 64, 'sw_fl_strides': 2,
          'sw_fl_kernel_size': 9}

train_data = [X_train, SW_train]
val_data = [X_val, SW_val]
test_data = [X_test[:, :, -mag_T0:], SW_test[:, -sw_T0:]]
hist, mod = models.train_cnn(train_data, [y_train, strength_train], val_data, [y_val, strength_val], params)

# EVALUATE MODELS
plt.figure()
plt.title("Loss")
plt.plot(hist.history['loss'], label='train')
plt.plot(hist.history['val_loss'], label='val')
plt.legend()

plt.figure()
plt.title("Accuracy")
plt.plot(hist.history['time_output_acc'], label='train')
plt.plot(hist.history['val_time_output_acc'], label='val')
plt.legend()


plt.figure()
plt.title("MAE")
plt.plot(hist.history['strength_output_mean_absolute_error'], label='train')
plt.plot(hist.history['val_strength_output_mean_absolute_error'], label='val')
plt.legend()

plt.figure()
plt.title("MSE")
plt.plot(hist.history['strength_output_mean_squared_error'], label='train')
plt.plot(hist.history['val_strength_output_mean_squared_error'], label='val')
plt.legend()

print("Evaluation: ", mod.evaluate(test_data, [y_test, strength_test]))

mod.save("saved models/{}.h5".format("StrengthNet"))

plt.show()
