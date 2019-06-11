import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.feature.nightshade import Nightshade
from matplotlib.collections import QuadMesh
import numpy as np


def plot_station_tracks(fig, gridspec, mag_data, ss_index=None, shading=None, names=None):

    lines = []
    labels = []
    share_ax = None
    for i in range(mag_data.shape[0]):
        if i == 0:
            ax = fig.add_subplot(gridspec[i])
            share_ax = ax
        else:
            ax = fig.add_subplot(gridspec[i], sharex=share_ax)
        lines.append(ax.plot(mag_data[i, :, 2])[0])
        lines.append(ax.plot(mag_data[i, :, 3])[0])
        lines.append(ax.plot(mag_data[i, :, 4])[0])

        if shading is not None:
            vmin = shading.min()
            vmax = shading.max()
            cam_max = max(0, 2 * np.nanmax(mag_data[i]))
            cam_min = min(0, 2 * np.nanmin(mag_data[i]))
            add_cam_shade(shading[i], cam_min, cam_max, vmin, vmax, ax)

        if ss_index is not None:
            ax.axvline(ss_index, linestyle='--', color='gray')

        if names is not None:
            ax.set_title(names[i])

        labels += ['N', 'E', 'Z']

        if i != mag_data.shape[0] - 1:
            ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        else:
            ax.set_xlabel("minutes")

    fig.legend(lines[:3], labels[:3])


def plot_solar_wind(fig, gridspec, data, shading=None, t0=64):

    b_ax = fig.add_subplot(gridspec)
    b_ax.set_title("Solar Wind")
    b_ax.set_ylabel("nT")
    b_ax.plot(data[:, 0], '--', label='Bx')
    b_ax.plot(data[:, 1], '--', label='By')
    b_ax.plot(data[:, 2], '--', label='Bz')
    b_ax.legend(loc='lower left')

    v_ax = b_ax.twinx()
    v_ax.set_ylabel("m/s")
    v_ax.plot(data[:, 3], '-', label='Vx')
    v_ax.plot(data[:, 4], '-', label='Vy')
    v_ax.plot(data[:, 5], '-', label='Vz')
    v_ax.legend(loc='lower right')

    if shading is not None:
        vmin = shading.min()
        vmax = shading.max()
        cam_max = max(0, 2 * np.nanmax(data))
        cam_min = min(0, 2 * np.nanmin(data))
        add_cam_shade(shading, cam_min, cam_max, vmin, vmax, b_ax, t0=t0)


def add_cam_shade(data, cam_min, cam_max, vmin, vmax, ax, t0=32):
    l = data.shape[0]
    cam_coords = np.stack((np.concatenate((np.arange(l + 1) + t0, np.arange(l + 1) + t0), axis=0),
                           np.concatenate((np.ones(l + 1) * cam_min, np.ones(l + 1) * cam_max), axis=0)), axis=1)
    cam_shade = QuadMesh(data.shape[0], 1, cam_coords, alpha=.25, array=data, cmap='coolwarm',
                         edgecolor=(1.0, 1.0, 1.0, 0.3), linewidth=0.0016)
    cam_shade.set_clim(vmin=vmin, vmax=vmax)
    ax.add_collection(cam_shade)


def plot_map_and_stations(fig, gridspec, station_locations, station_names, station_plot_dict, region_corners=None,
                          ss_location=None, date=None):
    map_ax = fig.add_subplot(gridspec, projection=ccrs.NearsidePerspective(-95, 57, 15000000))
    map_ax.stock_img()
    # Plot the border of region
    if region_corners is not None:
        map_ax.plot(np.linspace(region_corners[0][0], region_corners[1][0], 10), np.ones(10) * region_corners[0][1],
                    '--', color='gray', transform=ccrs.PlateCarree())

        map_ax.plot(np.ones(10) * region_corners[1][0], np.linspace(region_corners[0][1], region_corners[1][1], 10),
                    '--', color='gray', transform=ccrs.PlateCarree())

        map_ax.plot(np.linspace(region_corners[0][0], region_corners[1][0], 10), np.ones(10) * region_corners[1][1],
                    '--', color='gray', transform=ccrs.PlateCarree())

        map_ax.plot(np.ones(10) * region_corners[0][0], np.linspace(region_corners[0][1], region_corners[1][1], 10),
                    '--', color='gray', transform=ccrs.PlateCarree())

    # plot all stations
    map_ax.plot(station_locations[:, 0], station_locations[:, 1], '.', color='black', transform=ccrs.PlateCarree())
    # plot special stations
    for entry in station_plot_dict:
        map_ax.plot(station_locations[entry['indices'], 0], station_locations[entry['indices'], 1], '.',
                    color=entry['color'], transform=ccrs.PlateCarree(), label=entry['name'])
        if entry['station names']:
            for j in entry['indices']:
                map_ax.text(station_locations[j, 0] - 1, station_locations[j, 1] - 1, station_names[j],
                            horizontalalignment='right', transform=ccrs.PlateCarree())

    map_ax.legend()

    if ss_location is not None:
        map_ax.plot(ss_location[0], ss_location[1], 'rx', transform=ccrs.PlateCarree())

    if date is not None:
        map_ax.add_feature(Nightshade(date, alpha=0.2))


def plot_sme(sme, ss_index):
    # SME, SMU, -SML
    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
    ax[0].plot(sme[:, 0], label='SME')
    ax[0].plot(-1 * sme[:, 1], label='-1 * SML')
    ax[0].plot(sme[:, 2], label='SMU')
    ax[0].axvline(ss_index, linestyle='--')
    ax[0].set_ylabel("nT")
    ax[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax[0].legend()

    # location stuff
    ax[1].set_ylabel('MLAT')
    ax[1].plot(sme[:, 3], label='SML MLAT')
    ax[1].plot(sme[:, 4], label='SMU MLAT')
    ax[1].axvline(ss_index, linestyle='--')
    ax[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax[1].legend()

    ax[2].plot(sme[:, 5], label='SML MLT')
    ax[2].plot(sme[:, 6], label='SMU MLT')
    ax[2].set_ylabel('MLT')
    ax[2].axvline(ss_index, linestyle='--')
    ax[2].set_xlabel("minutes")
    ax[2].legend()


def plot_filter_output(index, station, layer=3):
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


def plot_cam(index):
    plt.figure()
    plt.pcolormesh(self.cams[index], cmap='coolwarm')


def confusion_mtx(y_true, y_pred):
    y_true = np.ravel(np.array(y_true))
    y_pred = np.ravel(np.array(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    return cm


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