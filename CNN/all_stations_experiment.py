"""

"""

import numpy as np
import keras
import matplotlib.pyplot as plt
import keras.backend as K
import modules
plt.style.use('ggplot')


def split_data(list_of_data, split, random=True):
    """this function splits a list of equal length (first dimension) data arrays into two lists. The length of the data
    put into the second list is determined by the 'split' argument. This can be used for slitting [X, y] into
    [X_train, y_train] and [X_val, y_val]
    """

    split_idx = int((1 - split) * list_of_data[0].shape[0])

    idx = np.arange(list_of_data[0].shape[0])
    if random:
        np.random.shuffle(idx)

    split_a = []
    split_b = []

    for data in list_of_data:
        split_a.append(data[idx[:split_idx]])
        split_b.append(data[idx[split_idx:]])

    return split_a, split_b


def true_positive(y_true, y_pred):
    y_pred_pos = K.round(y_pred[:, 0])
    y_pos = K.round(y_true[:, 0])
    return K.sum(y_pos * y_pred_pos) / (K.sum(y_pos) + K.epsilon())


def false_positive(y_true, y_pred):
    y_pred_pos = K.round(y_pred[:, 0])
    y_pos = K.round(y_true[:, 0])
    y_neg = 1 - y_pos
    return K.sum(y_pred_pos * y_neg) / (K.sum(y_neg) + K.epsilon())


# CONFIGURATION
batch_size = 64
epochs = 20
class_weight = {0: 1, 1: 1}
train_test_split = .2
train_val_split = .1
models = []
run_dict = {'station_conv': 1,
            'comb_net': 1,
            'resnet': 1}

# load in the data created by "create_dataset.py"
data = np.load("../../data/all_stations_data.npz")
X = data['X']
y = data['y'][:, None]

# create train, val and test sets
train, test = split_data([X, y], train_test_split, random=False)
train, val = split_data(train, train_val_split, random=True)
X_train, y_train = train
X_val, y_val = val
X_test, y_test = test

print("X train shape:", X_train.shape, "proportion of substorms: ", np.mean(y_train))
print("X val shape:", X_val.shape, "proportion of substorms: ", np.mean(y_val))
print("X test shape:", X_test.shape, "proportion of substorms: ", np.mean(y_test))

# CREATE MODELS
if run_dict['station_conv']:
    model_input = keras.layers.Input(shape=X_train.shape[1:])

    net = modules.conv_batch_relu(filters=64, kernel_size=[1, 11], strides=[1, 2])(model_input)
    net = modules.conv_batch_relu(filters=64, kernel_size=[1, 11], strides=[1, 2])(net)
    net = modules.conv_batch_relu(filters=128, kernel_size=[1, 11], strides=[1, 2])(net)
    net = modules.conv_batch_relu(filters=128, kernel_size=[1, 11], strides=[1, 2])(net)
    net = keras.layers.GlobalMaxPool2D()(net)

    net = keras.layers.Dense(1024, activation='relu')(net)
    model_output = keras.layers.Dense(1, activation='sigmoid')(net)

    station_conv_net = keras.models.Model(inputs=model_input, outputs=model_output)
    station_conv_net.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', true_positive, false_positive])
    station_conv_hist = station_conv_net.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                                             validation_data=(X_val, y_val))
    models.append({'name': 'Station Conv Net',
                   'hist': station_conv_hist,
                   'model': station_conv_net,
                   'test_data': (X_test, y_test),
                   'color': 'r'})


if run_dict['comb_net']:
    model_input = keras.layers.Input(shape=X_train.shape[1:])

    side_path = keras.layers.Conv2D(32, [X_train.shape[1], 1], strides=[1, 1], padding='valid')(model_input)
    side_path = keras.layers.BatchNormalization()(side_path)
    side_path = keras.layers.ReLU()(side_path)
    side_path_d = keras.layers.MaxPool2D([1, 2], strides=[1, 2])(side_path)

    net = keras.layers.Conv2D(32, [1, 5], strides=[1, 1], padding='same')(model_input)
    net = keras.layers.BatchNormalization()(net)
    net = keras.layers.ReLU()(net)

    net = keras.layers.Conv2D(32, [1, 5], strides=[1, 1], padding='same')(net)
    net = keras.layers.BatchNormalization()(net)
    net = keras.layers.ReLU()(net)

    net = keras.layers.MaxPool2D([1, 2], strides=[1, 2])(net)

    side_path = keras.layers.Conv2D(64, [X_train.shape[1], 1], strides=[1, 1], padding='valid')(net)
    side_path = keras.layers.BatchNormalization()(side_path)
    side_path = keras.layers.ReLU()(side_path)
    side_path = keras.layers.Concatenate()([side_path_d, side_path])
    side_path = keras.layers.Conv2D(64, [1, 1], strides=[1, 1], padding='valid')(side_path)
    side_path_d = keras.layers.MaxPool2D([1, 2], strides=[1, 2])(side_path)

    net = keras.layers.Conv2D(64, [1, 5], strides=[1, 1], padding='same')(net)
    net = keras.layers.BatchNormalization()(net)
    net = keras.layers.ReLU()(net)

    net = keras.layers.Conv2D(64, [1, 5], strides=[1, 1], padding='same')(net)
    net = keras.layers.BatchNormalization()(net)
    net = keras.layers.ReLU()(net)

    net = keras.layers.MaxPool2D([1, 2], strides=[1, 2])(net)

    net = keras.layers.Conv2D(128, [1, 5], strides=[1, 1], padding='same')(net)
    net = keras.layers.BatchNormalization()(net)
    net = keras.layers.ReLU()(net)

    net = keras.layers.Conv2D(128, [X_train.shape[1], 1], strides=[1, 1], padding='valid')(net)
    net = keras.layers.BatchNormalization()(net)
    net = keras.layers.ReLU()(net)

    net = keras.layers.Concatenate()([side_path_d, net])

    net = keras.layers.Conv2D(256, [1, 1], strides=[1, 1], padding='same')(net)
    net = keras.layers.BatchNormalization()(net)
    net = keras.layers.ReLU()(net)

    net = keras.layers.Conv2D(2, [1, 1], strides=[1, 1], padding='same')(net)
    net = keras.layers.BatchNormalization()(net)
    net = keras.layers.ReLU()(net)

    net = keras.layers.GlobalAveragePooling2D()(net)
    model_output = keras.layers.Softmax()(net)

    comb_net = keras.models.Model(inputs=model_input, outputs=net)
    comb_net.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy', true_positive, false_positive])
    comb_hist = comb_net.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                             validation_data=(X_val, y_val), class_weight=class_weight)

    models.append({'name': 'Combination Conv Net',
                   'hist': comb_hist,
                   'model': comb_net,
                   'test_data': (X_test, y_test),
                   'color': 'b'})

if run_dict['resnet']:
    resnet_input = keras.layers.Input(X_train.shape[1:])
    blocks = [2, 2, 2, 2]

    numerical_names = [True] * len(blocks)

    x = keras.layers.Conv2D(64, (1, 13), strides=(1, 1), use_bias=False, name="conv1", padding="same")(resnet_input)
    x = keras.layers.BatchNormalization(name="bn_conv1")(x)
    x = keras.layers.Activation("relu", name="conv1_relu")(x)
    x = keras.layers.MaxPooling2D((1, 3), strides=(2, 2), padding="same", name="pool1")(x)

    features = 64

    outputs = []

    for stage_id, iterations in enumerate(blocks):
        for block_id in range(iterations):
            x = modules.mag_block(
                features,
                stage_id,
                block_id,
                numerical_name=(block_id > 0 and numerical_names[stage_id]),
                freeze_bn=False
            )(x)

        features *= 2

        outputs.append(x)

    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(2, activation='softmax')(x)
    model = keras.models.Model(inputs=resnet_input, outputs=x)

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy', true_positive, false_positive])
    resnet_hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),
                            class_weight=class_weight)


    models.append({'name': 'Res Net',
                   'hist': resnet_hist,
                   'model': model,
                   'test_data': (X_test, y_test),
                   'color': 'g'})


# EVALUATE MODELS
plt.figure()
plt.title("Loss")
for model in models:
    plt.plot(model['hist'].history['loss'], model['color']+'-', label=model['name'] + ' train')
    plt.plot(model['hist'].history['val_loss'], model['color']+'--', label=model['name'] + ' val')
plt.legend()

plt.figure()
plt.title("Accuracy")
for model in models:
    plt.plot(model['hist'].history['acc'], model['color']+'-', label=model['name'] + ' train')
    plt.plot(model['hist'].history['val_acc'], model['color']+'--', label=model['name'] + ' val')
plt.legend()

plt.figure()
plt.title("True Positive")
for model in models:
    plt.plot(model['hist'].history['true_positive'], model['color']+'-', label=model['name'] + ' train')
    plt.plot(model['hist'].history['val_true_positive'], model['color']+'--', label=model['name'] + ' val')
plt.legend()

plt.figure()
plt.title("False Positive")
for model in models:
    plt.plot(model['hist'].history['false_positive'], model['color']+'-', label=model['name'] + ' train')
    plt.plot(model['hist'].history['val_false_positive'], model['color']+'--', label=model['name'] + ' val')
plt.legend()

for model in models:
    model['test_accuracy'] = model['model'].evaluate(model['test_data'][0], model['test_data'][1], batch_size=batch_size)[1]

for model in models:
    print(model['model'].summary())

for model in models:
    print(model['name'] + " Accuracy: ", model['test_accuracy'])

plt.show()
