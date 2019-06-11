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
from detection import utils


class SupermagDataset:

    def __init__(self, args, substorm_fn="substorms.csv", sme_fn="SME.csv", stations_fn="supermag_stations.csv",
                 solar_wind_fn="solar_wind.pkl", output_fn=None, mag_fn_pattern="mag_data/mag_data_{}.nc",
                 use_strength=False, regression=False):

        self.Tm = args.Tm  # length of interval to use as input data
        self.Tp = args.Tp  # length of prediction interval
        self.Tw = args.Tw  # minutes
        self.region_corners = [[-130, 45], [-60, 70]]
        self.sme_window = 20
        self.n_pos_classes = args.posclasses
        self.ex_per_ss = args.experss
        self.min_per_class = self.Tp / self.n_pos_classes

        self.output_fn = output_fn
        self.mag_fn_pattern = mag_fn_pattern
        if output_fn is None:
            if not regression:
                self.output_fn = "{}classes_data{}.npz".format(self.n_pos_classes + 1, self.Tm)
            else:
                self.output_fn = "{}classes_data{}.npz".format(self.Tm)

        self.substorms = self.open_csv(substorm_fn)

        self.use_strength = use_strength
        self.regression = regression

        self.sme = pd.read_csv(sme_fn, index_col=0)
        self.sme.index = pd.to_datetime(self.sme.index)

        self.solar_wind = pd.read_pickle(solar_wind_fn)

        self.stations, self.station_locations = self.open_and_filter_stations(stations_fn)

        # dataset stats
        self.total_substorms = 0
        self.n_out_of_region = 0
        self.n_no_mag_data = 0
        self.kept_storms = 0

    def run(self, training_years, test_years):
        # training data
        train_dict = self.assemble_over_range(training_years, test=False)

        # test data
        test_dict = self.assemble_over_range(test_years, test=True)

        train_dict, test_dict = self.preprocess_data(train_dict, test_dict)

        # figure out good ordering for the stations (rows)
        dists = utils.distance_matrix(self.station_locations)
        dists[np.isnan(dists)] = 0

        sa = utils.SimAnneal(self.station_locations, dists, stopping_iter=100000000)
        sa.anneal()
        train_dict['mag_data_train'] = train_dict['mag_data_train'][:, sa.best_solution, :, :]
        test_dict['mag_data_test'] = test_dict['mag_data_test'][:, sa.best_solution, :, :]
        self.stations = [self.stations[s] for s in sa.best_solution]
        self.station_locations = self.station_locations[sa.best_solution]

        # save the dataset
        np.savez(self.output_fn, **{**train_dict, **test_dict, 'stations': self.stations,
                                    'station_locations': self.station_locations})

        print("total storms: ", self.total_substorms)
        print("Number skipped because of storms out of region: ", self.n_out_of_region, self.n_out_of_region / self.total_substorms)
        print("Number skipped because of missing data: ", self.n_no_mag_data, self.n_no_mag_data / self.total_substorms)
        print("Number of storms kept: ", self.kept_storms, self.kept_storms / self.total_substorms)

    @staticmethod
    def preprocess_data(train_dict, test_dict):

        mag_data_train = train_dict['mag_data_train'][:, :, :, 2:]
        mag_data_train[np.isnan(mag_data_train)] = 0
        train_dict['mag_data_train'][:, :, :, 2:] = mag_data_train

        mag_data_test = test_dict['mag_data_test'][:, :, :, 2:]
        mag_data_test[np.isnan(mag_data_test)] = 0
        test_dict['mag_data_test'][:, :, :, 2:] = mag_data_test

        return train_dict, test_dict

    def assemble_over_range(self, years, test=False):
        mag_data_list = []
        sw_data_list = []
        y_list = []
        sme_data_list = []
        ss_interval_index_list = []
        ss_location_list = []
        ss_dates_list = []
        for yr in years:
            print("{}: {}".format(yr, ['Train', 'Test'][test]))
            mag_data, sw_data, y, sme_data, ss_interval_index, ss_location, ss_dates = self.process_data_for_year(yr, test)
            mag_data_list.append(mag_data)
            sw_data_list.append(sw_data)
            y_list.append(y)
            sme_data_list.append(sme_data)
            ss_interval_index_list.append(ss_interval_index)
            ss_location_list.append(ss_location)
            ss_dates_list.append(ss_dates)

        # concatenate all of the data lists
        mag_data = np.concatenate(mag_data_list, axis=0)
        sw_data = np.concatenate(sw_data_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        sme_data = np.concatenate(sme_data_list, axis=0)
        ss_interval_index = np.concatenate(ss_interval_index_list, axis=0)
        ss_location = np.concatenate(ss_location_list, axis=0)
        ss_dates = np.concatenate(ss_dates_list, axis=0)
        ext = ['_train', '_test'][test]
        data_dict = {
            'mag_data' + ext: mag_data,
            'sw_data' + ext: sw_data,
            'y' + ext: y,
            'sme_data' + ext: sme_data,
            'ss_interval_index' + ext: ss_interval_index,
            'ss_location' + ext: ss_location,
            'ss_dates' + ext: ss_dates}
        return data_dict

    def process_data_for_year(self, year, test=False):
        """
        Info for each example:
            Data:
                [x] Mag Input (including station location)
                [x] SW Input
                [x] label target
                [x] strength target
                [x] location target (-1 for negative examples)
            metadata:
                [x] interval index
                [x] substorm date / time
                [x] strength interval index / SME location
                [x] (n stations with data can be calculated later)
            stats about dataset:
                [x] n substorms considered
                [x] n substorms accepted
                [ ] n rejected because outside region
                [ ] n rejected because not enough data

        If test:
            - extra mag data

        Parameters
        ----------
        year

        Returns
        -------
        mag_data, sw_data, y, sme_data, ss_interval_index, ss_location, ss_dates

        """
        if not isinstance(year, str):
            year = str(year)

        ss, data, dates = supermag_dataset.get_data_for_year(year)

        # choose random interval indexes for the substorms, gather the magnetometer sequences
        # TODO: filter out substorms that aren't near midnight?
        ss_dates = ss.index
        self.total_substorms += ss_dates.shape[0]
        ss_interval_index = np.cumsum(np.random.randint(1, self.Tp//self.ex_per_ss, (ss_dates.shape[0], self.ex_per_ss)), axis=1)

        ### POSITIVE EXAMPLES ###
        # each of these produces the data chunks and masks for which storms should be included
        if test:
            mag_data_p, mag_mask = self.gather_mag_data(data, dates, ss_dates, self.Tm, end=2 * self.Tp,
                                                        ss_interval_index=ss_interval_index)
        else:
            mag_data_p, mag_mask = self.gather_mag_data(data, dates, ss_dates, self.Tm,
                                                        ss_interval_index=ss_interval_index)
        self.n_no_mag_data += (~mag_mask).sum()
        sw_data_p, sw_mask = self.gather_sw_data(ss_dates, ss_interval_index)
        sme_data_p, sme_mask = self.gather_sme_data(ss_dates, self.ex_per_ss, test=test)

        # produce the mask to remove substorms not in the region
        rcheck = self.region_check(ss['MLT'].values, data[:, :, 0], dates, ss_dates)
        self.n_out_of_region += (~rcheck).sum()
        # combine all the masks
        p_mask = mag_mask * sw_mask * sme_mask * rcheck
        self.kept_storms += p_mask.sum()
        # expand masks for n examples per substorm
        p_mask_ext = np.stack([p_mask] * self.ex_per_ss, axis=1).ravel()
        ss_dates_p = np.stack([ss_dates] * self.ex_per_ss, axis=1).ravel()[p_mask_ext]
        # filter the data
        mag_data_p = mag_data_p[p_mask_ext]  # Magnetometer Input
        sw_data_p = sw_data_p[p_mask_ext]  # Solar Wind Input

        sme_data_p = sme_data_p[p_mask_ext]  # Strength Output
        ss_interval_index_p = ss_interval_index.ravel()[p_mask_ext]  # Minutes Away
        ss_location_p = np.stack([ss[['MLT', 'MLAT']].values[p_mask]] * self.ex_per_ss).reshape((-1, 2))  # Substorm Location
        y_p = np.ones(np.sum(p_mask_ext), dtype=bool)  # Label Output

        n_ss_examples = p_mask_ext.sum()

        ### NEGATIVE EXAMPLES ###
        mlt = data[:, :, 0]
        finmask = np.isfinite(mlt)
        finmask[finmask] *= mlt[finmask] > 12
        mlt[finmask] -= 24
        # restrict negative examples to where the region is near midnight (masking in mag time)
        mask = np.logical_and(np.nanmin(mlt, axis=0) < 2, np.nanmax(mlt, axis=0) > -2)
        # restrict negative examples to where there isn't a substorm in the next Tp minutes
        ss_index = np.argwhere(np.in1d(dates, ss_dates))[:, 0]
        mask[np.ravel(ss_index[:, None] - np.arange(self.Tp)[None, :])] = False
        possible_dates = dates[mask]
        negative_dates = possible_dates[np.cumsum(np.random.randint(self.Tm, possible_dates.shape[0] // n_ss_examples,
                                                                    n_ss_examples))]

        if test:
            mag_data_n, mag_mask = self.gather_mag_data(data, dates, negative_dates, self.Tm, end=2 * self.Tp)
        else:
            mag_data_n, mag_mask = self.gather_mag_data(data, dates, negative_dates, self.Tm)
        sw_data_n, sw_mask = self.gather_sw_data(negative_dates)
        sme_data_n, sme_mask = self.gather_sme_data(negative_dates, test=test)
        # mask over examples
        n_mask = mag_mask * sw_mask * sme_mask
        ss_dates_n = negative_dates[n_mask]
        # filter the data
        mag_data_n = mag_data_n[n_mask]  # Magnetometer Input
        sw_data_n = sw_data_n[n_mask]  # Solar Wind Input
        sme_data_n = sme_data_n[n_mask]  # Strength Output
        ss_interval_index_n = np.ones(n_mask.sum()) * -1  # Minutes Away
        ss_location_n = np.ones((n_mask.sum(), 2)) * -1  # Substorm Location
        y_n = np.zeros(np.sum(n_mask), dtype=bool)  # Label Output

        # stack positive and negative examples
        mag_data = np.concatenate((mag_data_p, mag_data_n), axis=0)
        sw_data = np.concatenate((sw_data_p, sw_data_n), axis=0)
        y = np.concatenate((y_p, y_n), axis=0)
        sme_data = np.concatenate((sme_data_p, sme_data_n), axis=0)
        ss_interval_index = np.concatenate((ss_interval_index_p, ss_interval_index_n), axis=0)
        ss_location = np.concatenate((ss_location_p, ss_location_n), axis=0)
        ss_dates = np.concatenate((ss_dates_p, ss_dates_n), axis=0)

        return mag_data, sw_data, y, sme_data, ss_interval_index, ss_location, ss_dates

    @staticmethod
    def region_check(ss_mlt, mag_mlt, dates, ss_dates):
        # TODO: Use MLAT as well
        rcheck = np.in1d(ss_dates, dates)
        ss_index = np.ones(rcheck.shape[0], dtype=int) * -1
        ss_index[rcheck] = np.argwhere(np.in1d(dates, ss_dates))[:, 0]
        mag_mlt_windows = mag_mlt[:, ss_index[:, None] + np.arange(-10, 11)]
        span = np.nanmax(mag_mlt_windows, axis=(0, 2)) - np.nanmin(mag_mlt_windows, axis=(0, 2))
        finmask = np.isfinite(mag_mlt_windows)
        finmask[finmask] *= mag_mlt_windows[finmask] > 12
        mask = finmask * (span[None, :, None] > 12)
        mag_mlt_windows[mask] -= 24
        mask = (span > 12) * (ss_mlt > 12)
        ss_mlt[mask] -= 24
        rcheck *= (ss_mlt > np.nanmin(mag_mlt_windows, axis=(0, 2))) * (ss_mlt < np.nanmax(mag_mlt_windows, axis=(0, 2)))
        return rcheck

    @staticmethod
    def gather_mag_data(data, dates, ss_dates, Tm, end=0, ss_interval_index=0):
        mask = np.in1d(ss_dates, dates)
        ss_index = np.ones(mask.shape[0], dtype=int) * -1
        ss_index[mask] = np.argwhere(np.in1d(dates, ss_dates))[:, 0]
        mask *= ss_index > Tm
        mag_interval_end = np.ravel(ss_index[:, None] - ss_interval_index)  # n_substorms x ex_per_ss
        # MLT - MLAT - N - E - Z
        mag_data = np.transpose(data[:, mag_interval_end[:, None] + np.arange(-Tm, end)[None, :], :], (1, 0, 2, 3))
        return mag_data, mask

    def gather_sw_data(self, ss_dates, ss_interval_index=0):
        mask = np.in1d(ss_dates, self.solar_wind.index)
        ss_index = np.ones(mask.shape[0], dtype=int) * -1
        ss_index[mask] = np.argwhere(np.in1d(self.solar_wind.index, ss_dates))[:, 0]
        mask *= ss_index > self.Tw
        sw_interval_end = np.ravel(ss_index[:, None] - ss_interval_index)
        sw_data = self.solar_wind.values[sw_interval_end[:, None] + np.arange(-self.Tw, 0)[None, :], :]
        return sw_data, mask

    def gather_sme_data(self, ss_dates, ex_per_ss=1, test=False):
        mask = np.in1d(ss_dates, self.sme.index)
        ss_index = np.ones(mask.shape[0], dtype=int) * -1
        j = np.argwhere(np.in1d(self.sme.index, ss_dates))[:, 0]
        ss_index[mask] = j
        ss_index = np.stack([ss_index] * ex_per_ss, axis=1).ravel()
        if test:
            sme = self.sme.values
            extension = ss_index.max() + 2 * self.Tp + 1 - sme.shape[0]
            if extension > 0:
                sme = np.concatenate((sme, np.nan * np.ones((extension, sme.shape[1]))), axis=0)
            sme_data = sme[ss_index[:, None] + np.arange(-self.Tm, 2 * self.Tp)[None, :]]
        else:
            sme_data_idx = np.argmin(self.sme['SML'].values[ss_index[:, None] + np.arange(0, self.sme_window)[None, :]],
                                     axis=1)
            sme_data = self.sme.iloc[ss_index + sme_data_idx].values
        return sme_data, mask

    def get_data_for_year(self, year):
        # gather substorms for the year
        ss = self.substorms[year]
        # gather magnetometer data for the year
        mag_file = self.mag_fn_pattern.format(year)
        # get rid of extra columns / put the columns in the desired order
        dataset = xr.open_dataset(mag_file).sel(dim_1=['MLT', 'MLAT', 'N', 'E', 'Z'])
        stations_in_dset = [st for st in dataset]
        # grab the dates before turning it into a numpy array
        dates = pd.to_datetime(dataset.Date_UTC.values)
        # turn the data into a big numpy array
        # filter out all stations outside of the region
        data = np.ones((len(self.stations), dates.shape[0], 5)) * np.nan  # (stations x time x component)
        for i, st in enumerate(self.stations):
            if st in stations_in_dset:
                data[i] = dataset[st]

        return ss, data, dates

    @staticmethod
    def open_csv(fn, date_col='Date_UTC', drop_col=['Unnamed: 0', 'Date_UTC']):
        data = pd.read_csv(fn)
        data.index = pd.to_datetime(data[date_col])
        data = data.drop(columns=drop_col)
        return data

    def open_and_filter_stations(self, stations_fn):
        all_stations = pd.read_csv(stations_fn, index_col=0, usecols=[0, 1, 2, 5])
        station_locations = all_stations.values[:, :2]
        station_locations[station_locations > 180] -= 360
        region_mask = ((station_locations[:, 0] > self.region_corners[0][0]) *
                       (station_locations[:, 0] < self.region_corners[1][0]) *
                       (station_locations[:, 1] > self.region_corners[0][1]) *
                       (station_locations[:, 1] < self.region_corners[1][1]))
        stations = list(all_stations[region_mask].index)
        station_locations = station_locations[region_mask]
        return stations, station_locations


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Create a substorm dataset')
    parser.add_argument('--Tp', nargs=1, default=30, type=int, help="prediction interval")
    parser.add_argument('--Tm', nargs=1, default=128, type=int, help="Mag data input interval")
    parser.add_argument('--Tw', nargs=1, default=256, type=int, help="Solar wind data input interval")
    parser.add_argument('--posclasses', nargs=1, default=1, type=int, help="Number of positive classes")
    parser.add_argument('--experss', nargs=1, default=1, type=int, help="Examples per substorm")

    args = parser.parse_args()

    supermag_dataset = SupermagDataset(args)

    np.random.seed(111)

    supermag_dataset.run(range(1990, 2015), range(2015, 2019))
