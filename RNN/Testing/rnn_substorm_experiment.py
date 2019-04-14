import numpy as np
import matplotlib.pyplot as plt
import os.path
import utils

plt.style.use('ggplot')

#my_path = os.path.abspath(os.path.dirname(__file__))
#path = os.path.join(my_path, "../../data/all_stations_data.npz")
path = "D:\substorm-detection\data\"
data = np.load(path)
X = data['X']
y = data['y'][:, None]

# create train, val and test sets
train, test = utils.split_data([X, y], train_test_split, random=False)
train, val = utils.split_data(train, train_val_split, random=True)
X_train, y_train = train
X_val, y_val = val
X_test, y_test = test