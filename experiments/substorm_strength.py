"""
I want to see if I can get the strength of the substorms by looking at which station has the largest response. Also
I want to see if this is the station where the substorm is located. (I'm not actually sure what I mean by "largest
response", does this mean largest negative 'N' component? Largest overall magnitude? Largest over a time window or
largest at the single time index where the substorm occurs? Then it would be cool to see if there is correlation
between the substorm strength and its predictability.
"""

import numpy as np
import keras
from detection import utils
import matplotlib.pyplot as plt
plt.style.use("ggplot")

# ss1 = pd.read_csv("../data/substorms_1990_2000.csv", index_col=0)
# ss1.index = pd.to_datetime(ss1.index)
# ss2 = pd.read_csv("../data/substorms_2000_2018.csv", index_col=0)
# ss2.index = pd.to_datetime(ss2.index)
#
# ss = pd.concat((ss1, ss2))
#
# n_storms = []
# for year in range(1990, 2019):
#     n_storms.append(len(ss[str(year)]))
#
# plt.plot(n_storms)
# plt.show()

model_file = '../CNN/saved models/StrengthNet.h5'

data_fn = "../data/1classes_data64_withsw.npz"
data = np.load(data_fn)
X = data['X']
y = data['y'][:, None]
sw_data = data['SW']
strength = data['strength']

train_test_split = .11
train, test = utils.split_data([X, y, sw_data, strength], train_test_split, random=False)
X_train, y_train, sw_data_train, strength_train = train
X_test, y_test, sw_data_test, strength_test = test

model: keras.models.Model = keras.models.load_model(model_file, custom_objects={'true_positive': utils.true_positive,
                                                                                'false_positive': utils.false_positive})
pred = model.predict([X_test, sw_data_test[:, -240:]])

pos_mask = y_test[:, 0] == 1
logits = np.log(pred[0][pos_mask].max(axis=1) / (1 - pred[0][pos_mask].max(axis=1)))
st = strength_test[pos_mask]

err = st - pred[1][pos_mask, 0]
print(np.std(err))
print(np.mean(err))
plt.figure()
plt.hist(err, 100)

plt.figure()
plt.plot(strength_test[pos_mask], pred[1][pos_mask, 0], 'b.')
plt.plot(strength_test[~pos_mask], pred[1][~pos_mask, 0], 'r.')
plt.plot(np.arange(2000), 'k--')
plt.show()
