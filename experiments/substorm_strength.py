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
import keras
import utils
import matplotlib.pyplot as plt
plt.style.use("ggplot")

# window = 10

# substorm_fn = "../data/substorms_2000_2018.csv"
# mag_file = "../data/mag_data2/mag_data_2000.nc"

# substorms = pd.read_csv(substorm_fn, index_col=0)
# substorms.index = pd.to_datetime(substorms.index)
# substorms = substorms['2000']
#
# dataset = xr.open_dataset(mag_file).sel(dim_1=['MLT', 'MLAT', 'N', 'E', 'Z'])
# dates = pd.to_datetime(dataset.Date_UTC.values)
# dataset = dataset.to_array().values
# print(len(substorms))
#
# ss_idx = np.argwhere(np.in1d(dates, substorms.index))[:, 0]
# data = np.stack([dataset[:, i-window:i+window] for i in ss_idx], axis=0)
#
# strengths = -1 * np.nanmin(data[:, :, :, 2], axis=(1, 2))
# plt.hist(strengths, 100)
# plt.show()

model_file = '../CNN/saved models/Multi-Station Conv Net64.h5'

data_fn = "../data/all_stations_data_160.npz"
data = np.load(data_fn)
X = data['X']
y = data['y'][:, None]
sw_data = data['SW']
strength = data['strength']

train_test_split = .11
train, test = utils.split_data([X, y, sw_data, strength], train_test_split, random=False)
X_test, y_test, sw_data_test, strength_test = test

model: keras.models.Model = keras.models.load_model(model_file, custom_objects={'true_positive': utils.true_positive, 'false_positive': utils.false_positive})
y_pred = model.predict(X_test[:, :, -64:, :])

pos_mask = y_test[:, 0] == 1
logits = np.log(y_pred[pos_mask, 0] / (1 - y_pred[pos_mask, 0]))
st = strength_test[pos_mask]

plt.plot(np.log(st), logits, '.')
plt.title("Multi-Station Conv Net64: Test")
plt.xlabel("Log substorm strength")
plt.ylabel("output logit")
plt.show()
