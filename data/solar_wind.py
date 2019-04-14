"""
performs linear interpolation on the solar wind data set, combines them into a single pickle file
"""
import pandas as pd
import numpy as np

data_dir = "solar_wind/"
for yr in range(2000, 2019):
    data_fn = "solar_wind_{}.csv".format(yr)

    v = pd.read_csv(data_dir + data_fn, index_col=0)
    v.index = pd.to_datetime(v.index)
    for col in v:
        mask = v[col] != 999999
        v[col] = np.interp(v.index, v.index[mask], v[col][mask])

    if yr == 2000:
        data = v
    else:
        data = pd.concat((data, v))

print("data shape: ", data.shape)
print("Nans: ", np.isnan(data).sum())

data.to_pickle("solar_wind.pkl")
