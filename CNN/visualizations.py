from CNN import models
import utils
import numpy as np
import keras
from keras.utils import plot_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import pandas as pd
sns.set()

########################################################################################################################
# CONFIGURATION
########################################################################################################################
TRAIN = False

data_fn = "../data/2classes_data128_withsw_small.npz"
train_test_split = .11
train_val_split = .15
model_file = "saved models/final_cnn_model.h5"

params = {'batch_size': 8, 'epochs': 15, 'verbose': 2, 'n_classes': 2,
          'time_output_weight': 1000000, 'SW': True,

          'mag_T0': 96, 'mag_stages': 1, 'mag_blocks_per_stage': 4,
          'mag_downsampling_strides': (2, 3),
          'mag_kernel_size': (2, 11), 'mag_fl_filters': 16,
          'mag_fl_strides': (1, 3),
          'mag_fl_kernel_size': (1, 13), 'mag_type': 'basic',

          'sw_T0': 192, 'sw_stages': 1, 'sw_blocks_per_stage': 1,
          'sw_downsampling_strides': 4, 'sw_kernel_size': 7, 'sw_fl_filters': 16,
          'sw_fl_strides': 3, 'sw_fl_kernel_size': 15, 'sw_type': 'residual'}

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
y_true = y_true[:, 0].astype(int)

# CLASS ACTIVATION MAPS ################################################################################################
cams = utils.batch_cam(mod, test_data, 64, 32)
# don't include evidence from missing data
no_loc_mask = np.any(st_loc_test == -1, axis=-1)
# "attention" (softmax activation)
cams[no_loc_mask] = -999
attn = np.exp(cams) / np.sum(np.exp(cams), axis=(1, 2), keepdims=True)
cams[no_loc_mask] = 999
neg_attn = np.exp(-cams) / np.sum(np.exp(-cams), axis=(1, 2), keepdims=True)
cams[no_loc_mask] = 0

fig, ax = plt.subplots(3, 3)
fig.suptitle("Class Activation Maps")
for i in range(3):
    for j in range(3):
        num = i * 3 + j
        ax[i, j].pcolormesh(cams[num], vmin=cams[:9].min(), vmax=cams[:9].max(), cmap='RdBu_r')
        ax[i, j].set_title("P: {:4.2f}, L: {}, I: {}".format(y_pred[num], y_true[num], ind_test[num]))

# LOCATIONS ############################################################################################################
mask = np.all(st_loc_test != -1, axis=-1)
locs1 = np.sum(attn[:, :, :, None] * np.where(mask[:, :, :, None], st_loc_test, 0), axis=(1, 2))
locs2 = st_loc_test[np.arange(st_loc_test.shape[0]), np.argmax(np.max(attn, axis=2), axis=1), -1] + [.5, 0]
err1 = locs1[y_true == 1] - ss_loc_test[y_true == 1]
err1[err1[:, 0] > 12, 0] -= 24
err1[err1[:, 0] < -12, 0] += 24
err2 = locs2[y_true == 1] - ss_loc_test[y_true == 1]
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
    mask = (t * min_per_interval <= ind_test) * (ind_test < (t + 1) * min_per_interval)
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

# FEATURES #############################################################################################################
n_examples = 3
# true positive
tp_mask = (pred_lab == 1) * (y_true == 1)
attn_threshold = np.max(attn[tp_mask, :, :], axis=2).max(axis=1).min() * .9
# find mag tracks where attention is > threshold
tracks = np.argwhere(np.max(attn[tp_mask, :, :], axis=2) > attn_threshold)
for _ in range(n_examples):
    i = np.random.randint(0, tracks[:, 0].max())
    current_tracks = tracks[tracks[:, 0] == i, :]
    fig, ax = plt.subplots(current_tracks.shape[0], 1, sharex='col', sharey='row')
    fig.suptitle("True Positive Features")
    for j in range(current_tracks.shape[0]):
        station = current_tracks[j, 1]
        if current_tracks.shape[0] == 1:
            cur_ax = ax
        else:
            cur_ax = ax[j]
        cur_ax.set_title("Station {}".format(station))
        cur_ax.plot(X_test[tp_mask, station, :, 0][i], label='N')
        cur_ax.plot(X_test[tp_mask, station, :, 1][i], label='E')
        cur_ax.plot(X_test[tp_mask, station, :, 2][i], label='Z')
        for t in range(X_test.shape[2]):
            cur_ax.axvline(x=t, alpha=attn[tp_mask, station][i][t], linewidth=7.5)
        cur_ax.legend()

# true negative
tn_mask = (pred_lab == 0) * (y_true == 0)
attn_threshold = np.max(neg_attn[tn_mask, :, :], axis=2).max(axis=1).min() * .9
# find mag tracks where attention is > threshold
tracks = np.argwhere(np.max(attn[tn_mask, :, :], axis=2) > attn_threshold)
for _ in range(n_examples):
    i = np.random.randint(0, tracks[:, 0].max())
    current_tracks = tracks[tracks[:, 0] == i, :]
    fig, ax = plt.subplots(current_tracks.shape[0], 1, sharex='col', sharey='row')
    fig.suptitle("True Negative Features")
    for j in range(current_tracks.shape[0]):
        station = current_tracks[j, 1]
        if current_tracks.shape[0] == 1:
            cur_ax = ax
        else:
            cur_ax = ax[j]
        cur_ax.set_title("Station {}".format(station))
        cur_ax.plot(X_test[tn_mask, station, :, 0][i], label='N')
        cur_ax.plot(X_test[tn_mask, station, :, 1][i], label='E')
        cur_ax.plot(X_test[tn_mask, station, :, 2][i], label='Z')
        for t in range(X_test.shape[2]):
            cur_ax.axvline(x=t, alpha=neg_attn[tn_mask, station][i][t], linewidth=7.5)
        cur_ax.legend()

# false positive
fp_mask = (pred_lab == 1) * (y_true == 0)
attn_threshold = np.max(attn[fp_mask, :, :], axis=2).max(axis=1).min() * .9
# find mag tracks where attention is > threshold
tracks = np.argwhere(np.max(attn[fp_mask, :, :], axis=2) > attn_threshold)
for _ in range(n_examples):
    i = np.random.randint(0, tracks[:, 0].max())
    current_tracks = tracks[tracks[:, 0] == i, :]
    fig, ax = plt.subplots(current_tracks.shape[0], 1, sharex='col', sharey='row')
    fig.suptitle("False Positive Features")
    for j in range(current_tracks.shape[0]):
        station = current_tracks[j, 1]
        if current_tracks.shape[0] == 1:
            cur_ax = ax
        else:
            cur_ax = ax[j]
        cur_ax.set_title("Station {}".format(station))
        cur_ax.plot(X_test[fp_mask, station, :, 0][i], label='N')
        cur_ax.plot(X_test[fp_mask, station, :, 1][i], label='E')
        cur_ax.plot(X_test[fp_mask, station, :, 2][i], label='Z')
        for t in range(X_test.shape[2]):
            cur_ax.axvline(x=t, alpha=attn[fp_mask, station][i][t], linewidth=7.5)
        cur_ax.legend()

# false negative
fn_mask = (pred_lab == 0) * (y_true == 1)
attn_threshold = np.max(neg_attn[fn_mask, :, :], axis=2).max(axis=1).min() * .9
# find mag tracks where attention is > threshold
tracks = np.argwhere(np.max(neg_attn[fn_mask, :, :], axis=2) > attn_threshold)
for _ in range(n_examples):
    i = np.random.randint(0, tracks[:, 0].max())
    current_tracks = tracks[tracks[:, 0] == i, :]
    fig, ax = plt.subplots(current_tracks.shape[0], 1, sharex='col', sharey='row')
    fig.suptitle("False Negative Features")
    for j in range(current_tracks.shape[0]):
        station = current_tracks[j, 1]
        if current_tracks.shape[0] == 1:
            cur_ax = ax
        else:
            cur_ax = ax[j]
        cur_ax.set_title("Station {}".format(station))
        cur_ax.plot(X_test[fn_mask, station, :, 0][i], label='N')
        cur_ax.plot(X_test[fn_mask, station, :, 1][i], label='E')
        cur_ax.plot(X_test[fn_mask, station, :, 2][i], label='Z')
        for t in range(X_test.shape[2]):
            cur_ax.axvline(x=t, alpha=neg_attn[fn_mask, station][i][t], linewidth=7.5)
        cur_ax.legend()

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

plt.show()
