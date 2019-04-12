import numpy as np
import matplotlib.pyplot as plt
from CNN import models
import utils
plt.style.use('ggplot')


# CONFIGURATION
batch_size = 64
epochs = 10
class_weight = {0: 1, 1: 1}
train_test_split = .2
train_val_split = .1
run_dict = {'station_conv': 1,
            'comb_net': 1,
            'resnet': 0}
model_list = []

# load in the data created by "create_dataset.py"
data = np.load("../data/all_stations_data.npz")
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
if run_dict['station_conv']:
    params = {'stages': 3,
              'blocks_per_stage': 2,
              'kernel_width': 11,
              'downsampling_per_stage': 2,
              'batch_size': batch_size,
              'epochs': epochs,
              'flx2': True,
              'fl_filters': 32,
              'fl_stride': 2,
              'fl_kernel_width': 11}
    hist, mod = models.train_strided_station_cnn(X_train, y_train, X_val, y_val, params)
    model_list.append({'name': 'Station Conv Net',
                       'hist': hist,
                       'model': mod,
                       'test_data': (X_test, y_test),
                       'color': 'r'})


if run_dict['comb_net']:
    params = {'stages': 2,
              'blocks_per_stage': 2,
              'kernel_width': 9,
              'downsampling_per_stage': 2,
              'batch_size': batch_size,
              'epochs': epochs,
              'flx2': True,
              'fl_filters': 32,
              'fl_stride': 2,
              'fl_kernel_width': 11}
    hist, mod = models.train_combiner_net(X_train, y_train, X_val, y_val, params)
    model_list.append({'name': 'Combiner Net',
                       'hist': hist,
                       'model': mod,
                       'test_data': (X_test, y_test),
                       'color': 'b'})

if run_dict['resnet']:
    pass
    # model_list.append({'name': 'Res Net',
    #                'hist': resnet_hist,
    #                'model': model,
    #                'test_data': (X_test, y_test),
    #                'color': 'g'})


# EVALUATE MODELS
plt.figure()
plt.title("Loss")
for model in model_list:
    plt.plot(model['hist'].history['loss'], model['color']+'-', label=model['name'] + ' train')
    plt.plot(model['hist'].history['val_loss'], model['color']+'--', label=model['name'] + ' val')
plt.legend()

plt.figure()
plt.title("Accuracy")
for model in model_list:
    plt.plot(model['hist'].history['acc'], model['color']+'-', label=model['name'] + ' train')
    plt.plot(model['hist'].history['val_acc'], model['color']+'--', label=model['name'] + ' val')
plt.legend()

plt.figure()
plt.title("True Positive")
for model in model_list:
    plt.plot(model['hist'].history['true_positive'], model['color']+'-', label=model['name'] + ' train')
    plt.plot(model['hist'].history['val_true_positive'], model['color']+'--', label=model['name'] + ' val')
plt.legend()

plt.figure()
plt.title("False Positive")
for model in model_list:
    plt.plot(model['hist'].history['false_positive'], model['color']+'-', label=model['name'] + ' train')
    plt.plot(model['hist'].history['val_false_positive'], model['color']+'--', label=model['name'] + ' val')
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
