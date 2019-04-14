import keras
from tensorflow import errors
from CNN import blocks
import utils


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
    try:
        hist = model.fit(X_train[:, :, -params['T0']:], y_train, batch_size=params['batch_size'], epochs=params['epochs'],
                         validation_data=(X_val[:, :, -params['T0']:], y_val), verbose=params['verbose'])
    except errors.ResourceExhaustedError:
        params = {'T0': 32,
                  'stages': 1,
                  'blocks_per_stage': 1,
                  'kernel_width': 5,
                  'downsampling_per_stage': 2,
                  'batch_size': 100,
                  'epochs': 1,
                  'flx2': True,
                  'fl_filters': 1,
                  'fl_stride': 2,
                  'fl_kernel_width': 5}
        return train_strided_station_cnn(X_train[:100], y_train[:100], X_val[:100], y_val[:100], params)

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
    try:
        hist = model.fit(X_train[:, :, -params['T0']:], y_train, batch_size=params['batch_size'], epochs=params['epochs'],
                         validation_data=(X_val[:, :, -params['T0']:], y_val), verbose=params['verbose'])
    except errors.ResourceExhaustedError:
        print("OUT OF MEMORY: training mini model instead, just ignore this one")
        params = {'T0': 32,
                  'stages': 1,
                  'blocks_per_stage': 1,
                  'batch_size': 100,
                  'epochs': 1,
                  'flx2': False,
                  'kernel_size': [1, 3],
                  'downsampling_strides': [1, 2],
                  'fl_filters': 1,
                  'fl_strides': [1, 2],
                  'fl_kernel_size': [1, 5],
                  'verbose': 2}
        return train_strided_multistation_cnn(X_train[:100], y_train[:100], X_val[:100], y_val[:100], params)

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