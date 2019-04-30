"""
This script is used to create a dataset where the input is  a time series from all stations. It goes
through every year of the overall dataset and creates training examples for both the positive and negative classes.

For the positive class, it selects a substorm and chooses a random "interval index" for the substorm. This number
is the time index (0-60) within the hour long prediction interval at which the substorm takes place. This is so the
network has examples of predicting substorms scattered throughout the hour long prediction interval. It selects the
2 hours of magnetometer data before the prediction interval and creates an input example:
x = (all stations x 128 minutes of mag data x 3 magnetic field components), y = 1.

OBJECTIVES:
    - 0/1 substorm/no substorm
    - multiclass for [no substorm, substorm in first 5 min, next 5 min, etc...]
    - strength (maximum SME over a window)

For the negative classes, just select a time index which has no substorm.

NOTE: NaNs are replaced with zeros
"""

import numpy as np
import pandas as pd
import xarray as xr
from pymap3d.vincenty import vdist
import anneal

use_swind = True

T0 = 128  # length of interval to use as input data
Tfinal = 30  # length of prediction interval
swind_T0 = 256  # minutes
region_corners = [[-130, 45], [-60, 70]]
window = 20
n_pos_classes = 1
ex_per_ss = 3
min_per_class = Tfinal / n_pos_classes

substorm_fn = "substorms.csv"
sme_fn = "SME.csv"
stations_fn = "supermag_stations.csv"
solar_wind_fn = "solar_wind.pkl"
output_fn = "{}classes_data{}_{}.npz".format(n_pos_classes+1, T0, ['withoutsw', 'withsw'][use_swind])
mag_fn_pattern = "mag_data/mag_data_{}.nc"

# open substorm file, make it datetime indexable
substorms = pd.read_csv(substorm_fn)
substorms.index = pd.to_datetime(substorms['Date_UTC'])
substorms = substorms.drop(columns=['Unnamed: 0', 'Date_UTC'])

# open SME file
sme = pd.read_csv(sme_fn)
sme.index = pd.to_datetime(sme['Date_UTC'])
sme = sme.drop(columns=['Date_UTC', 'Unnamed: 0'])
sme['SME'][sme['SME'] > 10000] = 0

# open stations file, filter out stations not in the correct region
all_stations = pd.read_csv(stations_fn, index_col=0, usecols=[0, 1, 2, 5])
station_locations = all_stations.values[:, :2]
station_locations[station_locations > 180] -= 360
region_mask = ((station_locations[:, 0] > region_corners[0][0]) * (station_locations[:, 0] < region_corners[1][0]) *
               (station_locations[:, 1] > region_corners[0][1]) * (station_locations[:, 1] < region_corners[1][1]))
stations = list(all_stations[region_mask].index)
station_locations = station_locations[region_mask]

# open solar wind file
solar_wind = pd.read_pickle(solar_wind_fn)

# data containers
X = []
SW = []
y = []
strength = []
date_data = []

# collecting dataset stats
total_substorms = 0
n_out_of_region = 0
n_no_mag_data = 0
kept_storms = 0

# iterate over years
for yr in range(1990, 2019):
    print(yr)
    # buffer for this year's data, to be concatenated into a numpy array later
    X_yr = []
    SW_yr = []
    y_yr = []
    date_data_yr = []
    strength_yr = []
    year = str(yr)

    # gather substorms for the year
    ss = substorms[year]
    # gather magnetometer data for the year
    mag_file = mag_fn_pattern.format(year)
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

        total_substorms += 1

        # substorm location
        ss_loc = ss.iloc[i][["MLT", "MLAT"]].astype(float)
        # substorm date
        date = np.datetime64(ss.index[i])
        # index within the entire year's worth of data that the substorm takes place
        ss_date_index = np.argmax(date == dates)
        # check if substorm within region
        region_check = data[:, ss_date_index, :2] - ss_loc[None, :]
        # filter out nans from check
        region_check = region_check[np.all(np.isfinite(region_check), axis=1)]
        # (doesn't go past the farthest station in region) (avoid converting to lat lon)
        if (np.all(region_check[:, 0] > 0) or np.all(region_check[:, 1] > 0) or
                np.all(region_check[:, 0] < 0) or np.all(region_check[:, 1] < 0)):
            n_out_of_region += 1
            continue

        # if the substorm occurs too early in the year (before T0 + substorm interval index), skip this substorm
        if ss_date_index - Tfinal - max(swind_T0, T0) < 0:
            print("Not enough mag data", ss.index[i], ss_date_index, ss_loc.ravel())
            n_no_mag_data += 1
            continue

        for j in range(ex_per_ss):
            # minute within the prediction interval at which the substorm takes place
            ss_interval_index = np.random.randint(1, Tfinal)

            # gather up the magnetometer data for the input interval
            mag_data = data[:, ss_date_index - ss_interval_index - T0:ss_date_index - ss_interval_index, 2:]

            if use_swind:
                # gather solar wind data
                sw_ss_index = np.argmax(date == solar_wind.index)
                sw_data = solar_wind.iloc[sw_ss_index - ss_interval_index - swind_T0:sw_ss_index - ss_interval_index]
                if sw_data.shape != (swind_T0, 6):
                    print("Bad SW: ", date)
                    continue
                SW_yr.append(sw_data.values)

            sme_idx = np.argmax(sme.index == ss.index[i])
            if sme_idx == 0:
                print("Bad SME: ", date)
                continue

            # add this example to this years data buffer
            X_yr.append(mag_data)
            y_yr.append(ss_interval_index // min_per_class + 1)
            strength_yr.append(np.nanmax(sme.iloc[sme_idx:sme_idx + window].values[:, 0]))

        kept_storms += 1

    # make sure to create equal number of positive and negative examples
    n_positive_examples = len(X_yr) // n_pos_classes
    print("{} substorms from {}".format(n_positive_examples, yr))
    while len(X_yr) < (n_pos_classes + 1) * n_positive_examples:
        # choose a random data during the year
        random_date_index = np.random.randint(max(swind_T0, T0) + Tfinal, dates.shape[0] - Tfinal)
        # skip this one if there is a substorm occurring (we are looking for negative examples here)
        if len(ss.iloc[random_date_index: random_date_index+Tfinal]) != 0:
            continue
        # collect the magnetometer data for this interval
        mlt = data[:, random_date_index - T0:random_date_index, 0]
        mlt = mlt[np.isfinite(mlt)]
        region_check = np.logical_or(mlt > 18, mlt < 6).sum(axis=0).min()
        if not region_check:
            continue
        mag_data = data[:, random_date_index - T0:random_date_index, 2:]
        if use_swind:
            offset = np.argmax(dates[random_date_index] == solar_wind.index)
            sw_data = solar_wind.iloc[offset - swind_T0:offset].values
            if sw_data.shape != (swind_T0, 6):
                print("Bad SW: ", dates[random_date_index])
                continue
            SW_yr.append(sw_data)
        # add the negative examples to this years data buffer
        X_yr.append(mag_data)
        y_yr.append(0)
        sme_idx = np.argmax(dates[random_date_index] == sme.index)
        strength_yr.append(np.nanmax(sme.iloc[sme_idx:sme_idx + window].values[:, 0]))

    # add this years data buffer to the overall data buffers
    X.append(np.stack(X_yr, axis=0))
    y.append(np.array(y_yr))
    if use_swind:
        SW.append(np.stack(SW_yr, axis=0))
    strength.append((np.array(strength_yr)))

# concatenate all of the data buffers into one big numpy array
X = np.concatenate(X, axis=0)
mask = ~np.all(np.isnan(X), axis=(0, 2, 3))
X = X[:, mask, :, :]
X[np.isnan(X)] = 0
y = np.concatenate(y, axis=0)
strength = np.concatenate(strength, axis=0)
if use_swind:
    SW = np.concatenate(SW, axis=0)

# figure out good ordering for the stations (rows)
station_locations = station_locations[mask]
a = station_locations[:, None, :] * np.ones((1, station_locations.shape[0], station_locations.shape[1]))
locs = np.reshape(np.concatenate((a, np.transpose(a, [1, 0, 2])), axis=2), (-1, 4)).astype(float)

d, a1, a2 = vdist(locs[:, 1], locs[:, 0], locs[:, 3], locs[:, 2])
dists = d.reshape((station_locations.shape[0], station_locations.shape[0]))
dists[np.isnan(dists)] = 0

sa = anneal.SimAnneal(station_locations, dists, stopping_iter=100000000)
sa.anneal()
X = X[:, sa.best_solution, :, :]

# save the dataset
if use_swind:
    np.savez(output_fn, X=X, y=y, SW=SW, strength=strength)
else:
    np.savez(output_fn, X=X, y=y, strength=strength)

print("total storms: ", total_substorms)
print("Number skipped because of storms out of region: ", n_out_of_region, n_out_of_region / total_substorms)
print("Number skipped because of missing data: ", n_no_mag_data, n_no_mag_data / total_substorms)
print("Number of storms kept: ", kept_storms, kept_storms / total_substorms)
