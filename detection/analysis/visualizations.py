"""
TODO:
    - invesitgate location predictions
    - why does sml still get very low even for non substorm cases:
        - check out what is going on during large sml, non substorm cases
    - is the prediction from one station or multiple?
    - trace activations through network
    - extract example patches of input which activate certain filters
    - gradient descent to find input which maximizes activation, play as a movie?
    - look at a particular layer:
        - look at two spatially seperate neurons in the same layer, find examples from dataset that highly
            activate these neurons, try and see which feature is translated


email 1:
Collect activations from every relu output in the network across all test set examples. Look for examples in the dataset
that produce high activations with certain filters. Extract the patch of the input that corresponds to this high
activation. Is this a single station or multiple stations? View individual station tracks over time as well as a movie
of the field vector. Will neurons be activated for similar inputs? Will the neurons activate for different looking
inputs?
Plots:
    - Globe + a few station tracks
    - Pcolormesh all stations, one component
    - Movie, rotating globe, similar to supermag website (can I just make a request to the website and download the
        actual movie!?!)

email 2:
Maximum activation input -> visualize with movie of the stations

email 3:
Find gradient with respect to CAM, multiply by CAM. Find gradient with respect to class, multiply by CAM. Look at
Grad-CAM, grad-cam ++
"""
import numpy as np
import keras
import keras.backend as K
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import transform
import pandas as pd

from detection.CNN import models
from detection import utils
from detection.analysis import plotting

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

    def get_layer_output(self, layer, x, batch_size=64):
        if isinstance(layer, int):
            layer = self.model.layers[layer]
        elif isinstance(layer, keras.layers.Layer):
            pass
        else:
            raise Exception("layer argument must be an integer layer number or a keras layer")

        layer_output_func = K.function(self.model.inputs, [layer.output])
        output_shape = [s.value for s in layer.output.shape]
        output_shape[0] = x[0].shape[0]
        layer_output = np.empty(output_shape)
        for i in range(int(np.ceil(output_shape[0] / batch_size))):
            cbs = min(batch_size, output_shape[0] - i * batch_size)
            x_cb = [u[i * batch_size:i * batch_size + cbs] for u in x]
            output_cb = layer_output_func(x_cb)[0]
            layer_output[i * batch_size:i * batch_size + cbs] = output_cb
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

    def get_important_stations(self, index, return_cam=False, grad=True):
        cam = self.mag_cams[index]

        if not grad:
            activated_stations = np.argsort(np.max(cam, axis=1))[::-1]
            if return_cam:
                return activated_stations, cam
            return activated_stations

        mag_grad = self.get_gradients(self.model.output[0], self.model.input[0],
                                      [x[None, index] for x in self.test_data], self.test_data[0].shape[1:3])[0]
        mag_grad_cam = np.maximum(cam[:, :, None] * mag_grad, 0).sum(axis=-1)
        activated_stations = np.argsort(np.max(mag_grad_cam, axis=1))[::-1]

        if return_cam:
            return activated_stations, mag_grad_cam

        return activated_stations

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
        activated_stations, mag_grad_cam = self.get_important_stations(index, return_cam=True)
        activated_stations = activated_stations[:3]

        # closest stations
        closest_stations = self.get_closest_stations_by_ind(ss_station_ind, index)
        # don't overlap with activated stations
        closest_stations = closest_stations[~np.in1d(closest_stations, activated_stations)][:3]

        # stations without data
        stations_without_data = np.mean(np.isnan(self.mag_data[index, :, self.mag_input_slice, 0]), axis=1) > .5

        stations_to_plot = np.concatenate((activated_stations, closest_stations), axis=0)
        station_plot_dict = [
            {'name': 'Activated Stations', 'indices': activated_stations, 'station names': True, 'color': 'blue'},
            {'name': 'Closest Stations', 'indices': closest_stations[~np.in1d(closest_stations, activated_stations)],
             'station names': True, 'color': 'green'},
            {'name': 'no data', 'indices': stations_without_data, 'station names': False, 'color': 'gray'}]

        sw_cont = np.sum(self.get_layer_output(28, [x[None, index] for x in self.test_data])[0] *
                         self.model.layers[31].get_weights()[0][:32, 0])
        mag_cont = np.sum(self.get_layer_output(29, [x[None, index] for x in self.test_data])[0] *
                          self.model.layers[31].get_weights()[0][32:, 0])

        # sw grad cam
        sw_grad = self.get_gradients(self.model.output[0], self.model.input[1], [x[None, index] for x in self.test_data])[0]
        sw_grad_cam = np.maximum(self.sw_cams[index, :, None] * sw_grad, 0).sum(axis=-1)

        fig = plt.figure(constrained_layout=True, figsize=(18, 14))
        fig.suptitle("Date: {}, SW Cont: {:5.3f}, Mag Cont: {:5.3f}".format(self.ss_dates[index], sw_cont, mag_cont))
        gs = fig.add_gridspec(nrows=7, ncols=4)
        plotting.plot_map_and_stations(fig, gs[:6, :2], self.station_locations, self.stations, station_plot_dict,
                                       self.region_corners, ss_location, self.ss_dates[index])

        plotting.plot_station_tracks(fig, [gs[j, 2:] for j in range(6)], self.mag_data[index, stations_to_plot], self.ss_index[index],
                                     mag_grad_cam[stations_to_plot], self.stations[stations_to_plot])

        plotting.plot_solar_wind(fig, gs[6, 1:3], self.sw_data[index], sw_grad_cam)

    def make_sme_plot(self, index):
        plotting.plot_sme(self.sme_data[index], 128)
