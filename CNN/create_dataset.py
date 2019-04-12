"""
This script is used to create a dataset where the input is  a time series from all stations. It goes
through every year of the overall dataset and creates training examples for both the positive and negative classes.

For the positive class, it selects a substorm and chooses a random "interval index" for the substorm. This number
is the time index (0-60) within the hour long prediction interval at which the substorm takes place. This is so the
network has examples of predicting substorms scattered throughout the hour long prediction interval. It selects the
2 hours of magnetometer data before the prediction interval and creates an input example:
x = (all stations x 128 minutes of mag data x 3 magnetic field components), y = 1.

The script also creates 2 time related labels, the first is a scalar and represents the portion of the way through
the hour long interval the substorm occurs (0 - 1), and the other treats each minute of the prediction interval as
a separate class and one-hot encodes the substorm time.

For the negative classes, just select a time index which has no substorm.

NOTE: NaNs are replaced with zeros
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
import xarray as xr
from pymap3d.vincenty import vdist
import os
os.chdir("C:\\Users\\Greg\\code\\space-physics-machine-learning")


T0 = 96  # length of interval to use as input data (~2 hours)
Tfinal = 10  # length of prediction interval
region_corners = [[-130, 45], [-60, 70]]

# substorm file, make it datetime indexable
substorms = pd.read_csv("data/substorms_2000_2018.csv")
substorms.index = pd.to_datetime(substorms.Date_UTC)

all_stations = pd.read_csv("data/supermag_stations.csv", index_col=0, usecols=[0, 1, 2, 5])
statloc = all_stations.values[:, :2]
statloc[statloc > 180] -= 360
region_mask = ((statloc[:, 0] > region_corners[0][0]) * (statloc[:, 0] < region_corners[1][0]) *
               (statloc[:, 1] > region_corners[0][1]) * (statloc[:, 1] < region_corners[1][1]))
stations = list(all_stations[region_mask].index)

X = []
y = []
for yr in range(2000, 2019):
    print(yr)
    # buffer for this year's data, to be concatenated into a numpy array later
    X_yr = []
    y_yr = []
    year = str(yr)

    # gather substorms for the year
    ss = substorms[year]
    # gather magnetometer data for the year
    mag_file = "./data/mag_data_{}.nc".format(year)
    # get rid of extra columns / put the columns in the desired order
    dataset = xr.open_dataset(mag_file).sel(dim_1=['MLT', 'MLAT', 'N', 'E', 'Z'])
    stations_in_dset = [st for st in dataset]
    # grab the dates before turning it into a numpy array
    dates = dataset.Date_UTC.values
    # turn the data into a big numpy array
    # filter out all stations outside of the region
    data = np.ones((len(stations), dates.shape[0], 5)) * np.nan  # (stations x time x component)
    for i, st in enumerate(stations):
        if st in stations_in_dset:
            data[i] = dataset[st]

    # find substorms
    for i in range(ss.shape[0]):
        # substorm location
        ss_loc = ss.iloc[i][["MLT", "MLAT"]].astype(float)
        # substorm date
        date = np.datetime64(ss.index[i])
        # minute within the prediction interval at which the substorm takes place
        ss_interval_index = np.random.randint(0, Tfinal)
        # index within the entire year's worth of data that the substorm takes place
        ss_date_index = np.argmax(date == dates)
        # if the substorm occurs too early in the year (before 2 hours + substorm interval index), skip this substorm
        if ss_date_index - ss_interval_index - T0 < 0:
            print("Not enough mag data", ss.index[i], ss_date_index, ss_loc.ravel())
            continue
        # check if substorm within region
        region_check = data[:, ss_date_index, :2] - ss_loc[None, :]
        # filter out nans from check
        region_check = region_check[np.all(np.isfinite(region_check), axis=1)]
        # (doesn't go past the farthest station in region) (avoid converting to lat lon)
        if (np.all(region_check[:, 0] > 0) or np.all(region_check[:, 1] > 0) or
                np.all(region_check[:, 0] < 0) or np.all(region_check[:, 1] < 0)):
            continue

        # gather up the magnetometer data for the input interval
        mag_data = data[:, ss_date_index - ss_interval_index - T0:ss_date_index - ss_interval_index, 2:]

        # add this example to this years data buffer
        X_yr.append(mag_data)
        y_yr.append(1)

    # make sure to create equal number of positive and negative examples
    n_positive_examples = len(X_yr)
    print("{} substorms from {}".format(n_positive_examples, yr))
    i = 0
    while i < n_positive_examples:
        # choose a random data during the year
        random_date_index = np.random.randint(T0 + Tfinal, dates.shape[0] - Tfinal)
        # skip this one if there is a substorm occurring (we are looking for negative examples here)
        if len(ss.iloc[random_date_index: random_date_index+Tfinal]) != 0:
            continue
        # collect the magnetometer data for this interval
        mag_data = data[:, random_date_index - T0:random_date_index, 2:]
        # add the negative examples to this years data buffer
        X_yr.append(mag_data)
        y_yr.append(0)
        i += 1

    # add this years data buffer to the overall data buffers
    X.append(np.stack(X_yr, axis=0))
    y.append(np.array(y_yr))

# concatenate all of the data buffers into one big numpy array
X = np.concatenate(X, axis=0)
X = X[:, ~np.all(np.isnan(X), axis=(0, 2, 3)), :, :]
X[np.isnan(X)] = 0
y = np.concatenate(y, axis=0)

# save the dataset
np.savez("./data/all_stations_data.npz", X=X, y=y)
