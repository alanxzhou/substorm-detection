import matplotlib.pyplot as plt
import pandas as pd
import cartopy.crs as ccrs
import utils
from cartopy.feature.nightshade import Nightshade
from matplotlib.collections import QuadMesh
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import numpy as np
import keras.backend as K
from skimage import transform


class MagGeoPlotter:
    """
    TODO:
        - SME / SMU / SML / locations of those quantities
        - first layer / other layer activations
    """
    def __init__(self, mag_data, sw_data, y, sme, ss_interval_index, ss_locations, ss_dates, stations,
                 station_locations, model, region_corners=[[-130, 45], [-60, 70]], Tm=96, Tw=192, Tp=30):
        self.mag_data = mag_data
        self.sw_data = sw_data
        self.y = y
        self.sme = sme  # SME, SML, SMU, SML_MLAT, SMU_MLAT, SML_MLT, SMU_MLT, SME_NUMSTATIONS, SMR_NUMSTATIONS
        self.ss_interval_index = ss_interval_index
        self.ss_locations = ss_locations
        self.ss_dates = pd.to_datetime(ss_dates)
        self.stations = stations
        self.station_locations = station_locations
        self.model = model
        self.region_corners = region_corners
        self.Tm = Tm
        self.Tp = Tp
        self.Tw = Tw
        self.t0 = self.mag_data.shape[2] - 2 * self.Tp
        self.ss_index = self.ss_interval_index + self.t0
        self.mag_input_slice = slice(self.t0 - self.Tm, self.t0)
        self.n_examples = self.mag_data.shape[0]
        self.n_stations = self.mag_data.shape[1]
        self.distance_matrix = utils.distance_matrix(self.station_locations)

        self.cams = self.batch_cam(64, 32)
        no_loc_mask = np.any(np.isnan(self.mag_data[:, :, self.mag_input_slice, :2]), axis=-1)
        self.cams[no_loc_mask] = 0

    def plot_filter_output(self, index, station, layer=3):
        filter_func = K.function(self.model.inputs, [self.model.layers[layer].output])
        n_filters = self.model.layers[layer].output.shape[-1].value
        x = [self.mag_data[index, None, :, self.mag_input_slice, 2:],
             self.sw_data[index, None, -self.Tw:]]
        activation = transform.resize(filter_func(x)[0][0], (self.n_stations, self.Tm, n_filters))[station]
        """
        a = np.zeros(96)
        for i in range(32):
            a += np.correlate(self.model.layers[23].get_weights()[0][0, :, i, 0], activation[:, i], 'same')
            plt.plot(np.correlate(self.model.layers[23].get_weights()[0][0, :, i, 0], activation[:, i], 'same'), 'b', alpha=.25)
        plt.plot(a, 'b')
        """

        plt.figure()
        plt.suptitle("Station: {}, Substorm: {}, Layer: {}".format(self.stations[station], index, layer))
        for i in range(n_filters):
            ax = plt.subplot(int(np.ceil(n_filters / 2)), 2, i + 1)
            ax.plot(np.arange(self.t0 - self.Tm, self.t0), activation[:, i])
            ax.set_ylim(activation.min() - 1, activation.max() + 1)
            vmin = self.cams[index, station].min()
            vmax = self.cams[index, station].max()
            self._plot_cam_shade(index, station, -20, 20, vmin, vmax, ax)
            if (i % 2) == 1:
                ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
            if i not in [n_filters - 1, n_filters - 2]:
                ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            else:
                ax.set_xlabel("minutes")

    def plot_sme(self, index):
        # SME, SMU, -SML
        plt.figure()
        plt.suptitle("{}".format(index))
        plt.subplot(311)
        plt.plot(self.sme[index, :, 0], label='SME')
        plt.plot(-1 * self.sme[index, :, 1], label='-1 * SML')
        plt.plot(self.sme[index, :, 2], label='SMU')
        plt.axvline(self.ss_index[index], linestyle='--')
        plt.ylabel("nT")
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        plt.legend()

        # location stuff
        plt.subplot(312)
        plt.ylabel('MLAT')
        plt.plot(self.sme[index, :, 3], label='SML MLAT')
        plt.plot(self.sme[index, :, 4], label='SMU MLAT')
        plt.axvline(self.ss_index[index], linestyle='--')
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        plt.legend()
        plt.subplot(313)
        plt.plot(self.sme[index, :, 5], label='SML MLT')
        plt.plot(self.sme[index, :, 6], label='SMU MLT')
        plt.ylabel('MLT')
        plt.axvline(self.ss_index[index], linestyle='--')
        plt.xlabel("minutes")
        plt.legend()

    def plot_cam(self, index):
        plt.figure()
        plt.pcolormesh(self.cams[index], cmap='coolwarm')

    def plot_map_with_mag_data(self, index):
        """
        Plot map with mag data on the side. Plot the two most activated stations and the two closest stations to the
        actual substorm. If there is overlap between those, then plot the next closest stations. Plot all the station
        locations on the map, highlight and add the name of the plotted stations.

        Parameters
        ----------
        index: which storm index to plot
        """
        # substorm location
        mlt_match = np.mod(abs(self.mag_data[index, :, :, 0] - self.ss_locations[index, 0]), 24) == 0
        mlat_match = abs(self.mag_data[index, :, :, 1] - self.ss_locations[index, 1]) == 0
        both_match = mlt_match * mlat_match
        ss_station_ind = np.argwhere(both_match)[0, 0]
        ss_location = self.station_locations[ss_station_ind]
        ## figure out which stations to plot ##
        # two most activated stations
        activated_stations = np.argsort(np.max(self.cams[index], axis=1))[-1:-4:-1]
        # two closest stations
        # only take stations with at least half data
        stations_without_data = np.mean(np.isnan(self.mag_data[index, :, self.mag_input_slice, 0]), axis=1) > .5
        dm = self.distance_matrix.copy()
        dm[ss_station_ind, stations_without_data] = 999999999
        closest_stations = np.argsort(dm[ss_station_ind])
        # don't overlap with the activated stations
        closest_stations = closest_stations[~np.in1d(closest_stations, activated_stations)][:3]
        stations_to_plot = np.concatenate((activated_stations, closest_stations), axis=0)

        fig = plt.figure()
        fig.suptitle("{} {} {} {}".format(self.ss_dates[index], self.ss_locations[index, 0], self.ss_locations[index, 1], self.ss_interval_index[index]))

        # map plotting
        station_plot_dict = [{'name': 'Activated Stations', 'indices': activated_stations, 'station names': True, 'color': 'blue'},
                             {'name': 'Closest Stations', 'indices': closest_stations, 'station names': True, 'color': 'green'},
                             {'name': 'no data', 'indices': stations_without_data, 'station names': False, 'color': 'gray'}]
        map_ax = fig.add_subplot(1, 2, 1, projection=ccrs.NearsidePerspective(-95, 57, 15000000))
        map_ax.stock_img()
        # stations
        self.plot_stations(map_ax, station_plot_dict)
        map_ax.plot(ss_location[0], ss_location[1], 'rx', transform=ccrs.PlateCarree())
        map_ax.add_feature(Nightshade(self.ss_dates[index], alpha=0.2))

        lines = []
        labels = []
        for i, s in enumerate(stations_to_plot):
            ax = fig.add_subplot(len(stations_to_plot), 2, 2 * (i+1))
            lines.append(ax.plot(self.mag_data[index, s, :, 2])[0])
            lines.append(ax.plot(self.mag_data[index, s, :, 3])[0])
            lines.append(ax.plot(self.mag_data[index, s, :, 4])[0])
            vmin = self.cams[index, stations_to_plot].min()
            vmax = self.cams[index, stations_to_plot].max()
            cam_max = max(0, 2 * np.nanmax(self.mag_data[index, s, :, 2:]))
            cam_min = min(0, 2 * np.nanmin(self.mag_data[index, s, :, 2:]))
            self._plot_cam_shade(i, s, cam_min, cam_max, vmin, vmax, ax)
            labels += ['N', 'E', 'Z']
            ax.axvline(self.ss_index[index], linestyle='--', color='gray')
            ax.set_title(self.stations[s])
            if i != len(stations_to_plot) - 1:
                ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            else:
                ax.set_xlabel("minutes")

        fig.legend(lines[:3], labels[:3])

        return stations_to_plot

    def _plot_cam_shade(self, index, station, cam_min, cam_max, vmin, vmax, ax):
        cam_coords = np.stack((np.concatenate((np.arange(97) + 32, np.arange(97) + 32), axis=0),
                               np.concatenate((np.ones(97) * cam_min, np.ones(97) * cam_max), axis=0)), axis=1)
        cam_shade = QuadMesh(96, 1, cam_coords, alpha=.25, array=self.cams[index, station], cmap='coolwarm',
                             edgecolor=(1.0, 1.0, 1.0, 0.3), linewidth=0.0016)
        cam_shade.set_clim(vmin=vmin, vmax=vmax)
        ax.add_collection(cam_shade)

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

    def plot_stations(self, map_ax, station_plot_dict):
        # Plot the border of region
        map_ax.plot(np.linspace(self.region_corners[0][0], self.region_corners[1][0], 10),
                    np.ones(10) * self.region_corners[0][1], '--', color='gray', transform=ccrs.PlateCarree())

        map_ax.plot(np.ones(10) * self.region_corners[1][0],
                    np.linspace(self.region_corners[0][1], self.region_corners[1][1], 10), '--', color='gray',
                    transform=ccrs.PlateCarree())

        map_ax.plot(np.linspace(self.region_corners[0][0], self.region_corners[1][0], 10),
                    np.ones(10) * self.region_corners[1][1], '--', color='gray', transform=ccrs.PlateCarree())

        map_ax.plot(np.ones(10) * self.region_corners[0][0],
                    np.linspace(self.region_corners[0][1], self.region_corners[1][1], 10), '--', color='gray',
                    transform=ccrs.PlateCarree())
        # plot all stations
        map_ax.plot(self.station_locations[:, 0], self.station_locations[:, 1], '.', color='black', transform=ccrs.PlateCarree())
        # plot special stations
        for entry in station_plot_dict:
            map_ax.plot(self.station_locations[entry['indices'], 0], self.station_locations[entry['indices'], 1], '.',
                        color=entry['color'], transform=ccrs.PlateCarree(), label=entry['name'])
            if entry['station names']:
                for j in entry['indices']:
                    map_ax.text(self.station_locations[j, 0] - 1, self.station_locations[j, 1] - 1, self.stations[j],
                                horizontalalignment='right', transform=ccrs.PlateCarree())
        map_ax.legend()

    def batch_cam(self, batch_size, mag_channels):
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

    @staticmethod
    def confusion_mtx(y_true, y_pred):
        y_true = np.ravel(np.array(y_true))
        y_pred = np.ravel(np.array(y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        return cm

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        classes = classes[unique_labels(y_true, y_pred)]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        return ax