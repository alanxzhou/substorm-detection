import keras
from CNN import blocks
import utils
import numpy as np


def train_strided_station_cnn(X_train, y_train, X_val, y_val, params):
    """ The concept here is that substorms will be detectable from individual stations, so this network
    processes stations individually (kernel size 1 in station dimension) and uses global max pooling to check
    if any of the stations are seeing a substorm.
    params:
        stages
        blocks_per_stage
        downsampling_per_stage
        batch_size
        epochs
        flx2
        kernel_width
        fl_filters
        fl_stride
        fl_kernel_width
    """
    model_input = keras.layers.Input(shape=[X_train.shape[1], params['T0'], X_train.shape[3]])
    net = blocks.conv_batch_relu(filters=params['fl_filters'], kernel_size=[1, params['fl_kernel_width']],
                                 strides=[1, params['fl_stride']])(model_input)

    filters = params['fl_filters']
    if params['flx2']:
        filters *= 2

    for stage in range(params['stages']):
        for _ in range(params['blocks_per_stage'] - 1):
            net = blocks.conv_batch_relu(filters=filters, kernel_size=[1, params['kernel_width']], strides=[1, 1])(net)
        net = blocks.conv_batch_relu(filters=filters, kernel_size=[1, params['kernel_width']],
                                     strides=[1, params['downsampling_per_stage']])(net)
        filters *= 2

    net = keras.layers.GlobalMaxPool2D()(net)

    model_output = keras.layers.Dense(1, activation='sigmoid')(net)

    model = keras.models.Model(inputs=model_input, outputs=model_output)
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy', utils.true_positive, utils.false_positive])

    hist = model.fit(X_train[:, :, -params['T0']:], y_train, batch_size=params['batch_size'], epochs=params['epochs'],
                     validation_data=(X_val[:, :, -params['T0']:], y_val), verbose=params['verbose'])

    return hist, model


def train_strided_multistation_cnn(X_train, y_train, X_val, y_val, params):
    """ This network assumes that the stations are ordered such that stations nearby in position are nearby on
    the globe. This will be similar to the regular station_cnn except the kernel will span more than 1 station
    params:
        stages
        blocks_per_stage
        downsampling_strides
        batch_size
        epochs
        flx2
        kernel_size
        fl_filters
        fl_strides
        fl_kernel_size
    """
    final_filters = params['fl_filters'] * 2 ** (params['stages'] + params['flx2'])
    if final_filters > 1024:
        params['stages'] = int(10 - np.log2(params['fl_filters']))
    if params['T0'] == 32 and (params['stages'] + params['flx2']) > 5:
        params['T0'] = 2 ** (params['stages'] + params['flx2'])
    print(params)
    model_input = keras.layers.Input(shape=[X_train.shape[1], params['T0'], X_train.shape[3]])
    net = blocks.conv_batch_relu(filters=params['fl_filters'], kernel_size=params['fl_kernel_size'],
                                 strides=params['fl_strides'])(model_input)

    filters = params['fl_filters']
    if params['flx2']:
        filters *= 2

    for stage in range(params['stages']):
        for _ in range(params['blocks_per_stage'] - 1):
            net = blocks.conv_batch_relu(filters=filters, kernel_size=params['kernel_size'], strides=[1, 1])(net)
        net = blocks.conv_batch_relu(filters=filters, kernel_size=params['kernel_size'],
                                     strides=params['downsampling_strides'])(net)
        filters *= 2

    net = keras.layers.GlobalAveragePooling2D()(net)
    model_output = keras.layers.Dense(1, activation='sigmoid')(net)

    model = keras.models.Model(inputs=model_input, outputs=model_output)
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy', utils.true_positive, utils.false_positive])

    hist = model.fit(X_train[:, :, -params['T0']:], y_train, batch_size=params['batch_size'], epochs=params['epochs'],
                     validation_data=(X_val[:, :, -params['T0']:], y_val), verbose=params['verbose'])

    return hist, model


def train_strided_multistation_cnn_with_swdata(X_train, y_train, X_val, y_val, params):
    """ X_train: [Mag_data, SW_data]
    This network uses solar wind data. Basically, it's two networks, one for mag data, one for solar wind, both ending
    with global average pooling. The outputs from both networks are concatenated and passed through a softmax
    classifier.

    params:
        mag_T0
        mag_stages
        mag_blocks_per_stage
        mag_downsampling_strides
        mag_batch_size
        mag_epochs
        mag_flx2
        mag_kernel_size
        mag_fl_filters
        mag_fl_strides
        mag_fl_kernel_size

        sw_T0
        sw_stages
        sw_blocks_per_stage
        sw_downsampling_strides
        sw_batch_size
        sw_epochs
        sw_flx2
        sw_kernel_size
        sw_fl_filters
        sw_fl_strides
        sw_fl_kernel_size
    """
    print(params)
    mag_data, sw_data = X_train
    mag_data_val, sw_data_val = X_val

    # Mag Net
    mag_input = keras.layers.Input(shape=[mag_data.shape[1], params['mag_T0'], mag_data.shape[3]])
    mag_net = blocks.conv_batch_relu(filters=params['mag_fl_filters'], kernel_size=params['mag_fl_kernel_size'],
                                 strides=params['mag_fl_strides'])(mag_input)

    filters = params['mag_fl_filters']
    if params['mag_flx2']:
        filters *= 2

    for stage in range(params['mag_stages']):
        for _ in range(params['mag_blocks_per_stage'] - 1):
            mag_net = blocks.conv_batch_relu(filters=filters, kernel_size=params['mag_kernel_size'], strides=[1, 1])(mag_net)
        mag_net = blocks.conv_batch_relu(filters=filters, kernel_size=params['mag_kernel_size'],
                                     strides=params['mag_downsampling_strides'])(mag_net)
        filters *= 2
    mag_net = keras.layers.GlobalAveragePooling2D()(mag_net)

    # Solar Wind Net
    sw_input = keras.layers.Input(shape=[params['sw_T0'], sw_data.shape[2]])
    sw_net = blocks.conv_batch_relu(filters=params['sw_fl_filters'], kernel_size=params['sw_fl_kernel_size'],
                                     strides=params['sw_fl_strides'])(sw_input)

    filters = params['sw_fl_filters']
    if params['sw_flx2']:
        filters *= 2

    for stage in range(params['sw_stages']):
        for _ in range(params['sw_blocks_per_stage'] - 1):
            sw_net = blocks.conv_batch_relu(filters=filters, kernel_size=params['sw_kernel_size'], strides=[1, 1])(sw_net)
        sw_net = blocks.conv_batch_relu(filters=filters, kernel_size=params['sw_kernel_size'],
                                         strides=params['sw_downsampling_strides'])(sw_net)
        filters *= 2
    sw_net = keras.layers.GlobalAveragePooling2D()(sw_net)

    # Concatenate the two results, apply a dense layer
    concatenation = keras.layers.Concatenate()([sw_net, mag_net])
    model_output = keras.layers.Dense(1, activation='sigmoid')(concatenation)

    model = keras.models.Model(inputs=[mag_input, sw_input], outputs=model_output)
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy', utils.true_positive, utils.false_positive])

    train_data = [mag_data[:, :, -params['mag_T0']:], sw_data[:, -params['sw_T0']:]]
    val_data = [mag_data_val[:, :, -params['mag_T0']:], sw_data_val[:, -params['sw_T0']:]]
    hist = model.fit(train_data, y_train, batch_size=params['batch_size'], epochs=params['epochs'],
                     validation_data=(val_data, y_val), verbose=params['verbose'])

    return hist, model


def train_combiner_net(X_train, y_train, X_val, y_val, params):
    model_input = keras.layers.Input(shape=[X_train.shape[1], params['T0'], X_train.shape[3]])
    net = blocks.conv_batch_relu(filters=params['fl_filters'], kernel_size=[1, params['fl_kernel_width']],
                                 strides=[1, params['fl_stride']])(model_input)

    filters = params['fl_filters']
    if params['flx2']:
        filters *= 2

    for stage in range(params['stages']):
        for _ in range(params['blocks_per_stage'] - 1):
            net = blocks.combiner(X_train.shape[1], filters=filters, kernel_size=[1, params['kernel_width']],
                                  strides=[1, 1])(net)
        net = blocks.combiner(X_train.shape[1], filters=filters, kernel_size=[1, params['kernel_width']],
                              strides=[1, params['downsampling_per_stage']])(net)
        filters *= 2

    net = keras.layers.GlobalAveragePooling2D()(net)

    model_output = keras.layers.Dense(1, activation='sigmoid')(net)

    model = keras.models.Model(inputs=model_input, outputs=model_output)
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy', utils.true_positive, utils.false_positive])
    hist = model.fit(X_train[:, :, -params['T0']:], y_train, batch_size=params['batch_size'], epochs=params['epochs'],
                     validation_data=(X_val[:, :, -params['T0']:], y_val))

    return hist, model


def train_resnet():
    model_input = keras.layers.Input(shape=[X_train.shape[1], params['T0'], X_train.shape[3]])
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
            x = blocks.mag_block(
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

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy', true_positive, false_positive])
    resnet_hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),
                            class_weight=class_weight)