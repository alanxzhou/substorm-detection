"""
TODO:
    - invesitgate location predictions
    - Individial Time Series Invesitgation:
        - scale better
    - why does sml still get very low even for non substorm cases:
        - check out what is going on during large sml, non substorm cases
"""
from CNN import models
import utils
import numpy as np
import keras
from keras.utils import plot_model
import plotting
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import pandas as pd
import cartopy.crs as ccrs
sns.set()

########################################################################################################################
# CONFIGURATION
########################################################################################################################
TRAIN = False

data_fn = "../data/2classes_data128.npz"
train_val_split = .15
model_file = "saved models/final_cnn_model.h5"

t_tot = 128

params = {'batch_size': 8, 'epochs': 18, 'verbose': 2, 'n_classes': 2,
          'time_output_weight': 1000000, 'SW': True,

          'Tm': 96, 'mag_stages': 1, 'mag_blocks_per_stage': 4,
          'mag_downsampling_strides': (2, 3),
          'mag_kernel_size': (2, 11), 'mag_fl_filters': 16,
          'mag_fl_strides': (1, 3),
          'mag_fl_kernel_size': (1, 13), 'mag_type': 'basic',

          'Tw': 192, 'sw_stages': 1, 'sw_blocks_per_stage': 1,
          'sw_downsampling_strides': 4, 'sw_kernel_size': 7, 'sw_fl_filters': 16,
          'sw_fl_strides': 3, 'sw_fl_kernel_size': 15, 'sw_type': 'residual'}

########################################################################################################################
# DATA LOADING
########################################################################################################################
data = np.load(data_fn)
mag_data_train = data['mag_data_train']  # MLT, MLAT, N, E, Z
sw_data_train = data['sw_data_train']
y_train = data['y_train']
sme_data_train = data['sme_data_train']
ss_interval_index_train = data['ss_interval_index_train'].astype(int)
ss_location_train = data['ss_location_train']
ss_dates_train = data['ss_dates_train']
mag_data_test = data['mag_data_test']  # MLT, MLAT, N, E, Z
sw_data_test = data['sw_data_test']
y_test = data['y_test']
sme_data_test = data['sme_data_test']
ss_interval_index_test = data['ss_interval_index_test'].astype(int)
ss_location_test = data['ss_location_test']
ss_dates_test = data['ss_dates_test']
stations = data['stations']
station_locations = data['station_locations']

sml_test = -1 * np.min(sme_data_test[np.arange(sme_data_test.shape[0])[:, None], t_tot + ss_interval_index_test[:, None]
                                     + np.arange(20)[None, :], 1], axis=1)

# create train, val and test sets
train, val = utils.split_data([mag_data_train, sw_data_train, y_train, sme_data_train, ss_interval_index_train,
                               ss_location_train, ss_dates_train], train_val_split, random=True)
del data
del mag_data_train
del y_train
del sme_data_train
del sw_data_train
del ss_interval_index_train
del ss_location_train
del ss_dates_train

mag_data_train, sw_data_train, y_train, sme_data_train, ss_interval_index_train, ss_location_train, ss_dates_train = train
mag_data_val, sw_data_val, y_val, sme_data_val, ss_interval_index_val, ss_location_val, ss_dates_val = val

train_data = [mag_data_train[:, :, t_tot-params['Tm']:t_tot, 2:], sw_data_train[:, -params['Tw']:]]
train_targets = [y_train, -1 * sme_data_train[:, 1]]
val_data = [mag_data_val[:, :, t_tot-params['Tm']:t_tot, 2:], sw_data_val[:, -params['Tw']:]]
val_targets = [y_val, -1 * sme_data_val[:, 1]]

shuff_idx = np.arange(mag_data_test.shape[0])
np.random.shuffle(shuff_idx)

mag_data_test = mag_data_test[shuff_idx]
sw_data_test = sw_data_test[shuff_idx]
y_test = y_test[shuff_idx]
sme_data_test = sme_data_test[shuff_idx]
ss_interval_index_test = ss_interval_index_test[shuff_idx]
ss_location_test = ss_location_test[shuff_idx]
ss_dates_test = ss_dates_test[shuff_idx]
sml_test = sml_test[shuff_idx]

test_data = [mag_data_test[:, :, t_tot-params['Tm']:t_tot, 2:], sw_data_test[:, -params['Tw']:]]
test_targets = [y_test, sml_test]

print("mag data train shape:", mag_data_train.shape, "proportion of substorms: ", np.mean(y_train))
print("mag data val shape:", mag_data_val.shape, "proportion of substorms: ", np.mean(y_val))
print("mag data test shape:", mag_data_test.shape, "proportion of substorms: ", np.mean(y_test))

########################################################################################################################
# MODEL
########################################################################################################################

if TRAIN:

    hist, mod = models.train_cnn(train_data, train_targets, val_data, val_targets, params)
    mod.summary()
    keras.models.save_model(mod, model_file)
    plt.figure()
    plt.subplot(211)
    plt.plot(hist.history['val_time_output_acc'])
    plt.plot(hist.history['time_output_acc'])
    plt.subplot(212)
    plt.plot(hist.history['val_strength_output_mean_absolute_error'])
    plt.plot(hist.history['strength_output_mean_absolute_error'])
else:
    mod = keras.models.load_model(model_file,
                                  custom_objects={'true_positive': utils.true_positive,
                                                  'false_positive': utils.false_positive})
    mod.summary()

########################################################################################################################
# ANALYSIS
########################################################################################################################

y_pred, strength_pred = mod.predict(test_data)
pred_lab = np.round(y_pred).astype(int)
y_true, strength_true = test_targets

y_pred = y_pred[:, 0]
strength_pred = strength_pred[:, 0]
pred_lab = pred_lab[:, 0]


"""
fig, ax = plt.subplots(3, 3)
fig.suptitle("Class Activation Maps")
for i in range(3):
    for j in range(3):
        num = i * 3 + j
        ax[i, j].pcolormesh(cams[num], vmin=cams[:9].min(), vmax=cams[:9].max(), cmap='RdBu_r')
        ax[i, j].set_title("P: {:4.2f}, L: {}, I: {}".format(y_pred[num], y_true[num], ss_interval_index_test[num]))

# LOCATIONS ############################################################################################################
st_loc_test = mag_data_test[:, :, -params['Tm']:, :2]
mask = np.all(np.isfinite(st_loc_test), axis=-1)
locs1 = np.sum(attn[:, :, :, None] * np.where(mask[:, :, :, None], st_loc_test, 0), axis=(1, 2))
max_tracks = st_loc_test[np.arange(st_loc_test.shape[0]), np.argmax(np.max(attn, axis=2), axis=1)]
finite_ind = np.argwhere(np.isfinite(max_tracks).all(axis=2))
final_finite = finite_ind[:-1][np.diff(finite_ind[:, 0]) == 1, 1]
final_finite = np.concatenate((final_finite, [finite_ind[-1, 1]]))
locs2 = st_loc_test[np.arange(st_loc_test.shape[0]), np.argmax(np.max(attn, axis=2), axis=1), final_finite] + [.5, 0]
err1 = locs1[y_true == 1] - ss_location_test[y_true == 1]
err1[err1[:, 0] > 12, 0] -= 24
err1[err1[:, 0] < -12, 0] += 24
err2 = locs2[y_true == 1] - ss_location_test[y_true == 1]
err2[err2[:, 0] > 12, 0] -= 24
err2[err2[:, 0] < -12, 0] += 24
err_df = pd.DataFrame(np.concatenate((err1, err2), axis=1), columns=['MLT_1', 'MLAT_1', 'MLT_2', 'MLAT_2'])

# error plot FIX LABELING
g = sns.jointplot("MLT_1", "MLAT_1", data=err_df, color=sns.color_palette()[0], joint_kws={'s': 5}, marginal_kws={'kde': True})
g.ax_joint.scatter(err_df["MLT_2"], err_df["MLAT_2"], color=sns.color_palette()[1], s=5)
sns.distplot(err_df["MLT_2"], vertical=False, ax=g.ax_marg_x, color=sns.color_palette()[1], kde=True)
sns.distplot(err_df["MLAT_2"], vertical=True, ax=g.ax_marg_y, color=sns.color_palette()[1], kde=True)

# TIME DIFFERENCE ######################################################################################################
prediction_interval = 30
min_per_interval = 5
acc = []
r2 = []
fig, ax = plt.subplots(2, 3, sharex='col', sharey='row')
for t in range(prediction_interval // min_per_interval):
    mask = (t * min_per_interval <= ss_interval_index_test) * (ss_interval_index_test < (t + 1) * min_per_interval)
    acc.append(np.mean(np.round(y_pred[mask]) == y_true[mask]))
    r2.append(metrics.r2_score(strength_true[mask], strength_pred[mask]))
    ax[t // 3, t % 3].plot(strength_true[mask], strength_pred[mask], '.')
    ax[t // 3, t % 3].set_xlim(left=0)
    ax[t // 3, t % 3].set_ylim(bottom=0)
plt.tight_layout()
fig.text(0.5, 0.02, 'True Strength', ha='center')
fig.text(0.02, 0.5, 'Predicted Strength', va='center', rotation='vertical')

fig, ax1 = plt.subplots()
color = 'tab:blue'
ax1.set_xlabel('Minutes away')
ax1.set_ylabel('Accuracy', color=color)
ax1.plot(np.arange(0, prediction_interval, min_per_interval), acc, color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim([0, 1])

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:orange'
ax2.set_ylabel('R2 score', color=color)  # we already handled the x-label with ax1
ax2.plot(np.arange(0, prediction_interval, min_per_interval), r2, color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.grid(None)
ax2.set_ylim([0, 1])

fig.tight_layout()
"""
# FEATURES #############################################################################################################

n_examples = 2
# true positive
tp_mask = (pred_lab == 1) * (y_true == 1)
tp_examples = np.argwhere(tp_mask)[:n_examples, 0]

plotter = plotting.MagGeoPlotter(mag_data_test, sw_data_test, y_test, sme_data_test, ss_interval_index_test,
                                 ss_location_test, ss_dates_test, stations, station_locations, mod)

for i in tp_examples:
    plotted_stations = plotter.plot_map_with_mag_data(i)
    plotter.plot_cam(i)
    plotter.plot_sme(i)
    plotter.plot_filter_output(i, plotted_stations[0], layer=2)
    # plotter.plot_filter_output(i, plotted_stations[0], layer=27)

"""
# strength pred vs actual
# get the R scores on here
strength_df = pd.DataFrame(np.stack((strength_true, strength_pred, y_true), axis=1), columns=['Strength True', 'Predicted Strength', 'Substorm'])
g = sns.lmplot('Strength True', 'Predicted Strength', col='Substorm', hue='Substorm', data=strength_df, scatter_kws={'s': 2})
r2_ss = metrics.r2_score(strength_true[y_true == 1], strength_pred[y_true == 1])
r2_nss = metrics.r2_score(strength_true[y_true == 0], strength_pred[y_true == 0])
print(r2_nss, r2_ss)


cmat: plt.Axes = utils.plot_confusion_matrix(y_true, pred_lab, np.array(['No Substorm', 'Substorm']),
                                             normalize=True, title="Confusion Matrix")
cmat.grid(None)


plot_model(mod, to_file="saved models/final_cnn_model.png", show_shapes=True, show_layer_names=False)

print(mod.evaluate(test_data, test_targets))
"""
plt.show()
