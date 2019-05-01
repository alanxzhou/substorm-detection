import numpy as np
from CNN import models
import utils
import keras.backend as K
import sklearn.metrics as metrics
import pickle


# CONFIGURATION
n_classes = 2
data_fn = "../data/{}classes_data128_withsw_small.npz".format(n_classes)
results_fn = "search_results_{}classes.pkl".format(n_classes)
N = 100
train_test_split = .11
train_val_split = .15

# load in the data created by "create_dataset.py"
data = np.load(data_fn)
X = data['X']
y = data['y'][:, None]
strength = data['strength']
SW = data['SW']

# create train, val and test sets
train, test = utils.split_data([X, SW, y, strength], train_test_split, random=False)
del data
del X
del y
del strength
del SW
train, val = utils.split_data(train, train_val_split, random=True)
X_train, SW_train, y_train, strength_train = train
X_val, SW_val, y_val, strength_val = val
X_test, SW_test, y_test, strength_test = test

train_data = [X_train, SW_train]
val_data = [X_val, SW_val]
test_data = [X_test, SW_test]

print("X train shape:", X_train.shape, "proportion of substorms: ", np.mean(y_train))
print("X val shape:", X_val.shape, "proportion of substorms: ", np.mean(y_val))
print("X test shape:", X_test.shape, "proportion of substorms: ", np.mean(y_test))

kernel_sizes = [(1, 5), (1, 7), (1, 9), (1, 11),
                (2, 5), (2, 7), (2, 9), (2, 11),
                (3, 5), (3, 7), (3, 9), (3, 11),
                (4, 5), (4, 7), (4, 9)]

fl_kernel_sizes = [(1, 13), (1, 7), (1, 9), (1, 11),
                   (2, 13), (2, 7), (2, 9), (2, 11),
                   (3, 13), (3, 7), (3, 9), (3, 11),
                   (4, 7), (4, 9), (4, 11), (4, 13)]

params = {'batch_size': [8, 16, 32, 64], 'epochs': [15], 'verbose': [2], 'n_classes': [n_classes],
          'time_output_weight': [100, 1000, 10000, 100000, 1000000, 10000000], 'SW': [True, False],

          'mag_T0': [64, 96, 128], 'mag_stages': [1, 2, 3, 4], 'mag_blocks_per_stage': [1, 2, 3, 4],
          'mag_downsampling_strides': [(1, 2), (2, 2), (2, 3), (3, 2), (3, 3)],
          'mag_kernel_size': kernel_sizes, 'mag_fl_filters': [16, 32, 48, 64],
          'mag_fl_strides': [(1, 2), (2, 2), (2, 3), (3, 2), (3, 3)],
          'mag_fl_kernel_size': fl_kernel_sizes, 'mag_type': ['basic', 'residual'],

          'sw_T0': [64, 96, 128, 160, 192, 224, 256], 'sw_stages': [1, 2, 3, 4], 'sw_blocks_per_stage': [1, 2, 3, 4],
          'sw_downsampling_strides': [2, 3, 4], 'sw_kernel_size': [5, 7, 9, 11], 'sw_fl_filters': [16, 32, 48, 64],
          'sw_fl_strides': [2, 3, 4], 'sw_fl_kernel_size': [5, 7, 9, 11, 13, 15], 'sw_type': ['basic', 'residual']}

all_results = []
for _ in range(N):
    try:
        # set up
        _params = {}
        ii = []
        for p in params:
            i = np.random.randint(0, len(params[p]))
            ii.append(i)
            _params[p] = params[p][i]

        _params['mag_downsampling_strides'] = (min(_params['mag_downsampling_strides'][1],
                                                   _params['mag_kernel_size'][0]),
                                               _params['mag_downsampling_strides'][1])
        _params['mag_fl_strides'] = (min(_params['mag_fl_strides'][1],
                                         _params['mag_fl_kernel_size'][0]),
                                     _params['mag_downsampling_strides'][1])

        print()
        print()
        print(_)
        print(ii)
        print(_params)

        # run
        hist, mod = models.train_cnn(train_data, [y_train, strength_train], val_data, [y_val, strength_val], _params)

        # evaluate
        mag_T0 = _params['mag_T0']
        sw_T0 = _params['sw_T0']
        if _params['SW']:
            test_data = [X_test[:, :, -mag_T0:], SW_test[:, -sw_T0:]]
        else:
            test_data = X_test[:, :, -mag_T0:]
        evaluation = mod.evaluate(test_data, [y_test, strength_test])
        names = mod.metrics_names
        [y_pred, strength_pred] = mod.predict(test_data)
        if n_classes > 2:
            C = metrics.confusion_matrix(y_test[:, 0], np.argmax(y_pred, axis=1))
        else:
            C = metrics.confusion_matrix(y_test[:, 0], np.round(y_pred[:, 0]))
        regression_results = np.empty((_params['n_classes']+1, 2))
        regression_results[0, 0] = metrics.mean_squared_error(strength_test, strength_pred)
        regression_results[0, 1] = metrics.r2_score(strength_test, strength_pred)
        for c in range(_params['n_classes']):
            mask = y_test[:, 0] == c
            regression_results[c + 1, 0] = metrics.mean_squared_error(strength_test[mask], strength_pred[mask])
            regression_results[c + 1, 1] = metrics.r2_score(strength_test[mask], strength_pred[mask])

        # save
        result = dict()
        result['history'] = hist.history
        result['evaluation'] = {n: v for n, v in zip(names, evaluation)}
        result['confusion_matrix'] = C
        result['params'] = _params
        result['regression_results'] = regression_results
        all_results.append(result)
        print(result['evaluation'])
        print(result['confusion_matrix'])
        print(result['regression_results'])

        del mod
        del hist
        K.clear_session()

    except Exception as e:
        print("SKIPPING: {}".format(e))
        continue

    with open(results_fn, 'wb') as f:
        pickle.dump(all_results, f)
