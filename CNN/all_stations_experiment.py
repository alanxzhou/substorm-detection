import numpy as np
import matplotlib.pyplot as plt
from CNN import models
import utils
plt.style.use('ggplot')


# CONFIGURATION
data_fn = "../data/all_stations_data_128.npz"
batch_size = 32
epochs = 8
class_weight = {0: 1, 1: 1}
train_test_split = .1
train_val_split = .15
run_dict = {'station_net': 1,
            'combiner_net': 1,
            'multistation_net': 1,
            'resnet': 0}
model_list = []

# load in the data created by "create_dataset.py"
data = np.load(data_fn)
X = data['X']
y = data['y'][:, None]

# create train, val and test sets
train, test = utils.split_data([X, y], train_test_split, random=False)
train, val = utils.split_data(train, train_val_split, random=True)
X_train, y_train = train
X_val, y_val = val
X_test, y_test = test

print("X train shape:", X_train.shape, "proportion of substorms: ", np.mean(y_train))
print("X val shape:", X_val.shape, "proportion of substorms: ", np.mean(y_val))
print("X test shape:", X_test.shape, "proportion of substorms: ", np.mean(y_test))

# CREATE MODELS
for T0 in [32, 64, 96, 128]:
    if run_dict['station_net']:
        params = {'T0': T0,
                  'stages': 2,
                  'blocks_per_stage': 2,
                  'kernel_width': 9,
                  'downsampling_per_stage': 2,
                  'batch_size': batch_size,
                  'epochs': epochs,
                  'flx2': True,
                  'fl_filters': 32,
                  'fl_stride': 2,
                  'fl_kernel_width': 13}
        hist, mod = models.train_strided_station_cnn(X_train, y_train, X_val, y_val, params)
        model_list.append({'name': 'Station Conv Net' + str(T0),
                           'hist': hist,
                           'model': mod,
                           'test_data': (X_test[:, :, -T0:], y_test)})


    if run_dict['combiner_net']:
        params = {'T0': T0,
                  'stages': 3,
                  'blocks_per_stage': 1,
                  'kernel_width': 9,
                  'downsampling_per_stage': 2,
                  'batch_size': batch_size,
                  'epochs': epochs,
                  'flx2': True,
                  'fl_filters': 32,
                  'fl_stride': 2,
                  'fl_kernel_width': 13}
        hist, mod = models.train_combiner_net(X_train, y_train, X_val, y_val, params)
        model_list.append({'name': 'Combiner Net' + str(T0),
                           'hist': hist,
                           'model': mod,
                           'test_data': (X_test[:, :, -T0:], y_test)})

    if run_dict['multistation_net']:
        params = {'T0': T0,
                  'stages': 2,
                  'blocks_per_stage': 2,
                  'batch_size': batch_size,
                  'epochs': epochs,
                  'flx2': True,
                  'kernel_size': [3, 9],
                  'downsampling_strides': [1, 2],
                  'fl_filters': 32,
                  'fl_strides': [1, 2],
                  'fl_kernel_size': [3, 13]}
        hist, mod = models.train_strided_multistation_cnn(X_train, y_train, X_val, y_val, params)
        model_list.append({'name': 'Multi-Station Conv Net' + str(T0),
                           'hist': hist,
                           'model': mod,
                           'test_data': (X_test[:, :, -T0:], y_test)})

    if run_dict['resnet']:
        pass
        # model_list.append({'name': 'Res Net',
        #                'hist': resnet_hist,
        #                'model': model,
        #                'test_data': (X_test, y_test),
        #                'color': 'g'})


cmap = plt.get_cmap('tab20')
colors = cmap(np.linspace(0, 1, len(model_list)))
for c, model in zip(colors, model_list):
    model['color'] = c

# EVALUATE MODELS
plt.figure()
plt.title("Loss")
for model in model_list:
    plt.plot(model['hist'].history['loss'], '-', color=model['color'], label=model['name'] + ' train')
    plt.plot(model['hist'].history['val_loss'], '--', color=model['color'], label=model['name'] + ' val')
plt.legend()

plt.figure()
plt.title("Accuracy")
for model in model_list:
    plt.plot(model['hist'].history['acc'],  '-', color=model['color'], label=model['name'] + ' train')
    plt.plot(model['hist'].history['val_acc'],  '--', color=model['color'], label=model['name'] + ' val')
plt.legend()

plt.figure()
plt.title("True Positive")
for model in model_list:
    plt.plot(model['hist'].history['true_positive'],  '-', color=model['color'], label=model['name'] + ' train')
    plt.plot(model['hist'].history['val_true_positive'],  '--', color=model['color'], label=model['name'] + ' val')
plt.legend()

plt.figure()
plt.title("False Positive")
for model in model_list:
    plt.plot(model['hist'].history['false_positive'],  '-', color=model['color'], label=model['name'] + ' train')
    plt.plot(model['hist'].history['val_false_positive'],  '--', color=model['color'], label=model['name'] + ' val')
plt.legend()

for model in model_list:
    model['test_accuracy'] = model['model'].evaluate(model['test_data'][0], model['test_data'][1], batch_size=batch_size)[1]

for model in model_list:
    print(model['name'])
    print(model['model'].summary())
    print()

for model in model_list:
    print(model['name'] + " Accuracy: ", model['test_accuracy'])

plt.show()
