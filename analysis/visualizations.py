"""
TODO:
    - invesitgate location predictions
    - why does sml still get very low even for non substorm cases:
        - check out what is going on during large sml, non substorm cases
"""
import numpy as np
import keras
import keras.backend as K
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import transform
import pandas as pd

from CNN import models
import utils
from analysis import plotting

sns.set()


class Visualizer:
    
    def __init__(self, data_fn, params, Tp=30, train_model=False, train_val_split=.15,
                 model_file="saved models/final_cnn_model.h5", region_corners=[[-130, 45], [-60, 70]]):
        
        self.Tm = params['Tm']
        self.Tw = params['Tw']
        self.Tp = Tp
        
        self.load_data_and_model(data_fn, params, train_model, train_val_split, model_file)
        
        self.region_corners = region_corners
        self.ss_index = self.ss_interval_index + self.t0
        self.mag_input_slice = slice(self.t0 - self.Tm, self.t0)
        self.n_examples = self.mag_data.shape[0]
        self.n_stations = self.mag_data.shape[1]
        self.distance_matrix = utils.distance_matrix(self.station_locations)

        self.mag_cams = self.mag_batch_cam(64, 32)
        no_loc_mask = np.any(np.isnan(self.mag_data[:, :, self.mag_input_slice, :2]), axis=-1)
        self.mag_cams[no_loc_mask] = 0

        self.sw_cams = self.sw_batch_cam(64, 32)

        y_pred, strength_pred = self.model.predict(self.test_data)
        pred_lab = np.round(y_pred).astype(int)

        self.y_pred = y_pred[:, 0]
        self.pred_lab = pred_lab[:, 0]

    def load_data_and_model(self, data_fn, params, train_model, train_val_split, model_file):
        data = np.load(data_fn)
        mag_data_train = data['mag_data_train']  # MLT, MLAT, N, E, Z
        
        mag_data_test = data['mag_data_test']  # MLT, MLAT, N, E, Z
        sw_data_test = data['sw_data_test']
        y_test = data['y_test']
        sme_data_test = data['sme_data_test']
        ss_interval_index_test = data['ss_interval_index_test'].astype(int)
        ss_location_test = data['ss_location_test']
        ss_dates_test = pd.to_datetime(data['ss_dates_test'])
        self.stations = np.array(data['stations'])
        self.station_locations = data['station_locations']

        self.t0 = mag_data_train.shape[2]
        sml_windows = self.t0 + ss_interval_index_test[:, None] + np.arange(20)[None, :]
        sml_test = -1 * np.min(sme_data_test[np.arange(sme_data_test.shape[0])[:, None], sml_windows, 1], axis=1)

        del data
        del mag_data_train

        shuff_idx = np.arange(mag_data_test.shape[0])
        np.random.shuffle(shuff_idx)

        self.mag_data = mag_data_test[shuff_idx]
        self.sw_data = sw_data_test[shuff_idx]
        self.y = y_test[shuff_idx]
        # SME, SML, SMU, SML_MLAT, SMU_MLAT, SML_MLT, SMU_MLT, SME_NUMSTATIONS, SMR_NUMSTATIONS
        self.sme_data = sme_data_test[shuff_idx]
        self.ss_interval_index = ss_interval_index_test[shuff_idx]
        self.ss_locations = ss_location_test[shuff_idx]
        self.ss_dates = ss_dates_test[shuff_idx]
        self.sml = sml_test[shuff_idx]

        self.test_data = [self.mag_data[:, :, self.t0 - params['Tm']:self.t0, 2:], self.sw_data[:, -params['Tw']:]]
        self.test_targets = [self.y, self.sml]

        if train_model:
            sw_data_train = data['sw_data_train']
            y_train = data['y_train']
            sme_data_train = data['sme_data_train']
            ss_interval_index_train = data['ss_interval_index_train'].astype(int)
            ss_location_train = data['ss_location_train']
            ss_dates_train = data['ss_dates_train']

            # create train, val and test sets
            train, val = utils.split_data(
                [mag_data_train, sw_data_train, y_train, sme_data_train, ss_interval_index_train,
                 ss_location_train, ss_dates_train], train_val_split, random=True)

            del y_train
            del sme_data_train
            del sw_data_train
            del ss_interval_index_train
            del ss_location_train
            del ss_dates_train

            (mag_data_train, sw_data_train, y_train, sme_data_train, ss_interval_index_train, ss_location_train,
             ss_dates_train) = train
            mag_data_val, sw_data_val, y_val, sme_data_val, ss_interval_index_val, ss_location_val, ss_dates_val = val

            train_data = [mag_data_train[:, :, self.t0 - params['Tm']:self.t0, 2:], sw_data_train[:, -params['Tw']:]]
            train_targets = [y_train, -1 * sme_data_train[:, 1]]
            val_data = [mag_data_val[:, :, self.t0 - params['Tm']:self.t0, 2:], sw_data_val[:, -params['Tw']:]]
            val_targets = [y_val, -1 * sme_data_val[:, 1]]

            hist, self.model = models.train_cnn(train_data, train_targets, val_data, val_targets, params)
            self.model.summary()
            keras.models.save_model(self.model, model_file)
            plt.figure()
            plt.subplot(211)
            plt.plot(hist.history['val_time_output_acc'])
            plt.plot(hist.history['time_output_acc'])
            plt.subplot(212)
            plt.plot(hist.history['val_strength_output_mean_absolute_error'])
            plt.plot(hist.history['strength_output_mean_absolute_error'])

            print("mag data train shape:", mag_data_train.shape, "proportion of substorms: ", np.mean(y_train))
            print("mag data val shape:", mag_data_val.shape, "proportion of substorms: ", np.mean(y_val))
        else:
            self.model = keras.models.load_model(model_file,
                                                 custom_objects={'true_positive': utils.true_positive,
                                                                 'false_positive': utils.false_positive})
            self.model.summary()

        print("mag data test shape:", self.mag_data.shape, "proportion of substorms: ", np.mean(self.y))
            
    def get_gradients(self, df, dx, x, output_shape=None):
        grad_tensor = K.gradients(df, dx)
        sess = K.get_session()
        grad = sess.run(grad_tensor, {self.model.inputs[0]: x[0], self.model.inputs[1]: x[1]})[0]
        if output_shape is not None:
            return transform.resize(grad, (x[0].shape[0], output_shape[0], output_shape[1], grad.shape[3]))
        return grad

    def get_layer_output(self, layer, x):
        layer_output_func = K.function(self.model.inputs, [self.model.layers[layer].output])
        layer_output = layer_output_func(x)[0]
        return layer_output

    def argsort_distance_to_substorm(self, index):
        mag_mlt = self.mag_data[index, :, self.ss_index[index] + 1, 0]
        mag_mlt[mag_mlt > 12] -= 24
        mag_mlat = self.mag_data[index, :, self.ss_index[index] + 1, 1]
        ss_mlt = self.ss_locations[index, 0]
        ss_mlat = self.ss_locations[index, 1]
        if ss_mlt > 12:
            ss_mlt -= 24

        mlt_diff = np.mod(abs(mag_mlt - ss_mlt), 12)
        mlat_diff = abs(mag_mlat - ss_mlat)

        dists = mlt_diff + mlat_diff

        return np.argsort(dists)

    def sw_batch_cam(self, batch_size, sw_channels):
        dense_weights = self.model.get_layer('time_output').get_weights()[0][:sw_channels, 0]
        last_conv = K.function(self.model.inputs, [self.model.layers[26].output])
        cams = np.empty((self.n_examples, self.Tw))
        for i in range(int(np.ceil(self.n_examples / batch_size))):
            cbs = min(batch_size, self.n_examples - i * batch_size)
            x = [self.mag_data[i * batch_size:i * batch_size + cbs, :, self.mag_input_slice, 2:],
                 self.sw_data[i * batch_size:i * batch_size + cbs, -self.Tw:]]
            cam = np.sum(last_conv(x)[0] * dense_weights[None, None, :], axis=-1)
            cams[i * batch_size:i * batch_size + cbs] = transform.resize(cam, (cbs, self.Tw))
        return cams

    def mag_batch_cam(self, batch_size, mag_channels):
        dense_weights = self.model.get_layer('time_output').get_weights()[0][-mag_channels:, 0]
        last_conv = K.function(self.model.inputs, [self.model.layers[-6].output])
        cams = np.empty((self.n_examples, self.n_stations, self.Tm))
        for i in range(int(np.ceil(self.n_examples / batch_size))):
            cbs = min(batch_size, self.n_examples - i * batch_size)
            x = [self.mag_data[i * batch_size:i * batch_size + cbs, :, self.mag_input_slice, 2:],
                 self.sw_data[i * batch_size:i * batch_size + cbs, -self.Tw:]]
            cam = np.sum(last_conv(x)[0] * dense_weights[None, None, None, :], axis=-1)
            cams[i * batch_size:i * batch_size + cbs] = transform.resize(cam, (cbs, self.n_stations, self.Tm))
        return cams

    def get_substorm_location(self, index):
        mlt_match = np.mod(abs(self.mag_data[index, :, :, 0] - self.ss_locations[index, 0]), 24) == 0
        mlat_match = abs(self.mag_data[index, :, :, 1] - self.ss_locations[index, 1]) == 0
        both_match = mlt_match * mlat_match
        ss_station_ind = np.argwhere(both_match)[0, 0]
        ss_location = self.station_locations[ss_station_ind]
        return ss_location, ss_station_ind

    def get_closest_stations_by_ind(self, station_ind, index):
        # only take stations with at least half data
        stations_without_data = np.mean(np.isnan(self.mag_data[index, :, self.mag_input_slice, 0]), axis=1) > .5
        dm = self.distance_matrix.copy()
        dm[station_ind, stations_without_data] = 999999999
        closest_stations = np.argsort(dm[station_ind])
        return closest_stations

    def map_and_station_plot(self, index):
        """
        Plot map with mag data on the side. Plot the three most activated stations and the three closest stations to the
        actual substorm. If there is overlap between those, then plot the next closest stations. Plot all the station
        locations on the map, highlight and add the name of the plotted stations.

        Parameters
        ----------
        index: which storm index to plot
        """
        # substorm location
        ss_location, ss_station_ind = self.get_substorm_location(index)

        # most activated stations
        cam = self.mag_cams[index]
        mag_grad = self.get_gradients(self.model.output[0], self.model.input[0],
                                      [x[None, index] for x in self.test_data], self.test_data[0].shape[1:3])[0]
        mag_grad_cam = np.maximum(cam[:, :, None] * mag_grad, 0).sum(axis=-1)
        activated_stations = np.argsort(np.max(mag_grad_cam, axis=1))[-1:-4:-1]

        # closest stations
        closest_stations = self.get_closest_stations_by_ind(ss_station_ind, index)
        # don't overlap with activated stations
        closest_stations = closest_stations[~np.in1d(closest_stations, activated_stations)][:3]

        # stations without data
        stations_without_data = np.mean(np.isnan(self.mag_data[index, :, self.mag_input_slice, 0]), axis=1) > .5

        stations_to_plot = np.concatenate((activated_stations, closest_stations), axis=0)
        station_plot_dict = [
            {'name': 'Activated Stations', 'indices': activated_stations, 'station names': True, 'color': 'blue'},
            {'name': 'Closest Stations', 'indices': closest_stations, 'station names': True, 'color': 'green'},
            {'name': 'no data', 'indices': stations_without_data, 'station names': False, 'color': 'gray'}]

        sw_cont = np.sum(self.get_layer_output(28, [x[None, index] for x in self.test_data])[0] *
                         self.model.layers[31].get_weights()[0][:32, 0])
        mag_cont = np.sum(self.get_layer_output(29, [x[None, index] for x in self.test_data])[0] *
                          self.model.layers[31].get_weights()[0][32:, 0])

        # sw grad cam
        sw_grad = self.get_gradients(self.model.output[0], self.model.input[1], [x[None, index] for x in self.test_data])[0]
        sw_grad_cam = np.maximum(self.sw_cams[index, :, None] * sw_grad, 0).sum(axis=-1)

        fig = plt.figure(constrained_layout=True)
        fig.suptitle("Date: {}, SW Cont: {:5.3f}, Mag Cont: {:5.3f}".format(self.ss_dates[index], sw_cont, mag_cont))
        gs = fig.add_gridspec(nrows=7, ncols=4)
        plotting.plot_map_and_stations(fig, gs[:6, :2], self.station_locations, self.stations, station_plot_dict,
                                       self.region_corners, ss_location, self.ss_dates[index])

        plotting.plot_station_tracks(fig, [gs[j, 2:] for j in range(6)], self.mag_data[index, stations_to_plot], self.ss_index[index],
                                     mag_grad_cam[stations_to_plot], self.stations[stations_to_plot])

        plotting.plot_solar_wind(fig, gs[6, 1:3], self.sw_data[index], sw_grad_cam)

########################################################################################################################
# ANALYSIS
########################################################################################################################
"""
y_pred, strength_pred = mod.predict(test_data)
pred_lab = np.round(y_pred).astype(int)
y_true, strength_true = test_targets

y_pred = y_pred[:, 0]
strength_pred = strength_pred[:, 0]
pred_lab = pred_lab[:, 0]

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

# FEATURES #############################################################################################################

n_examples = 1
# true positive
tp_mask = (pred_lab == 1) * (y_true == 1)
tp_examples = np.argwhere(tp_mask)[:n_examples, 0]

plotter = plotting.MagGeoPlotter(mag_data_test, sw_data_test, y_test, sme_data_test, ss_interval_index_test,
                                 ss_location_test, ss_dates_test, stations, station_locations, mod)

for i in tp_examples:
    plotted_stations = plotter.plot_map_with_mag_data(i)
    plotter.plot_cam(i)
    plotter.plot_sme(i)
    grads = plotter.get_gradients(mod.output[0], mod.inputs[0], [t[None, i] for t in test_data])
    plt.figure()
    print(np.argsort(np.sum(plotter.cams[i, :, :, None] * grads[0], axis=(1, 2)))[::-1])
    # plotter.plot_filter_output(i, plotted_stations[0], layer=27)


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

if __name__ == "__main__":
    
    TRAIN = False

    data_fn = "../data/2classes_data128.npz"
    train_val_split = .15
    model_file = "saved models/final_cnn_model.h5"

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
    
    visualizer = Visualizer(data_fn, params, train_model=TRAIN, train_val_split=train_val_split, model_file=model_file)
    true_pos = (visualizer.pred_lab == 1) * (visualizer.y == 1)
    true_pos_ind = np.argwhere(true_pos)[:, 0]
    for i in true_pos_ind[:5]:
        visualizer.map_and_station_plot(i)

    plt.show()
