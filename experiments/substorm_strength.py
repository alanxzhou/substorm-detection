"""
I want to see if I can get the strength of the substorms by looking at which station has the largest response. Also
I want to see if this is the station where the substorm is located. (I'm not actually sure what I mean by "largest
response", does this mean largest negative 'N' component? Largest overall magnitude? Largest over a time window or
largest at the single time index where the substorm occurs? Then it would be cool to see if there is correlation
between the substorm strength and its predictability.
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

ss_index = 7
strength_window = 10

substorm_fn = "../data/substorms_2000_2018.csv"
mag_file = "../data/new mag_data/mag_data_2000.nc"

substorms = pd.read_csv(substorm_fn, index_col=0)
substorms.index = pd.to_datetime(substorms.index)
substorms = substorms['2000']

dataset = xr.open_dataset(mag_file).sel(dim_1=['MLT', 'MLAT', 'N', 'E', 'Z'])
dates = pd.to_datetime(dataset.Date_UTC.values)

print(len(substorms))
strengths = []
for ss_index in range(200):
    ss_date_index = np.argmax(dates == pd.Timestamp(substorms.index[ss_index]))
    data = dataset.isel(Date_UTC=slice(ss_date_index-strength_window, ss_date_index+strength_window)).sel(dim_1='N').to_array().values
    try:
        strengths.append(-np.nanmin(data))
    except ValueError as e:
        print("Missing Data")
        continue

plt.hist(strengths, 20)
plt.show()
