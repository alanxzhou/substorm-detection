import numpy as np
import matplotlib.pyplot as plt
from CNN import models
import utils
plt.style.use('ggplot')


# CONFIGURATION
data_fn = "../data/data64.npz"
batch_size = 16
epochs = 2
mag_T0 = 64
train_test_split = .11
train_val_split = .15

# load in the data created by "create_dataset.py"
data = np.load(data_fn)
X = data['X']
y = data['y'][:, None]
strength = data['strength']

# create train, val and test sets
train, test = utils.split_data([X, y, strength], train_test_split, random=False)
train, val = utils.split_data(train, train_val_split, random=True)
X_train, y_train, strength_train = train
X_val, y_val, strength_val = val
X_test, y_test, strength_test = test

print("X train shape:", X_train.shape, "proportion of substorms: ", np.mean(y_train))
print("X val shape:", X_val.shape, "proportion of substorms: ", np.mean(y_val))
print("X test shape:", X_test.shape, "proportion of substorms: ", np.mean(y_test))

params = {'T0': mag_T0,
          'stages': 3,
          'blocks_per_stage': 3,
          'batch_size': batch_size,
          'epochs': epochs,
          'flx2': False,
          'kernel_size': [3, 5],
          'downsampling_strides': [2, 2],
          'fl_filters': 128,
          'fl_strides': [2, 3],
          'fl_kernel_size': [2, 13],
          'n_classes': 4,
          'verbose': 2}
hist, mod = models.substorm_strength_network(X_train, [y_train, strength_train], X_val, [y_val, strength_val], params)

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

print("Evaluation: ", mod.evaluate(X_test, [y_test, strength_test]))

mod.save("saved models/{}.h5".format("StrengthNet"))

plt.show()
