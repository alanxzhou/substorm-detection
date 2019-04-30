import keras
from CNN import blocks
import utils
import keras.backend as K


def train_cnn(X_train, y_train, X_val, y_val, params):
    """ X_train: [Mag_data, SW_data]
    This network uses solar wind data. Basically, it's two networks, one for mag data, one for solar wind, both ending
    with global average pooling. The outputs from both networks are concatenated and passed through a softmax
    classifier.

    params = {'batch_size': batch_size, 'epochs': epochs, 'verbose': 2, 'n_classes': 1,

              'mag_T0': mag_T0, 'mag_stages': 3, 'mag_blocks_per_stage': 3, 'mag_downsampling_strides': [2, 2],
              'mag_flx2': False, 'mag_kernel_size': [3, 5], 'mag_fl_filters': 128, 'mag_fl_strides': [2, 3],
              'mag_fl_kernel_size': [2, 13],

              'sw_T0': sw_T0, 'sw_stages': 3, 'sw_blocks_per_stage': 1, 'sw_downsampling_strides': 2,
              'sw_flx2': True, 'sw_kernel_size': 9, 'sw_fl_filters': 32, 'sw_fl_strides': 2,
              'sw_fl_kernel_size': 13}
    """
    if params['n_classes'] < 2:
        raise Exception("Neet at least 2 classes")

    SW = False
    if isinstance(X_train, list):
        SW = True
        mag_data, sw_data = X_train
        mag_data_val, sw_data_val = X_val
    else:
        mag_data = X_train
        mag_data_val = X_val

    mag_input = keras.layers.Input(shape=[mag_data.shape[1], params['mag_T0'], mag_data.shape[-1]])
    if params['mag_type'] == 'residual':
        mag_net = _residual_2d_net(params['mag_fl_filters'], params['mag_fl_kernel_size'], params['mag_fl_strides'],
                                   params['mag_stages'], params['mag_blocks_per_stage'], params['mag_kernel_size'],
                                   params['mag_downsampling_strides'])(mag_input)
    elif params['mag_type'] == 'basic':
        mag_net = _basic_2d_net(params['mag_fl_filters'], params['mag_fl_kernel_size'], params['mag_fl_strides'],
                                params['mag_stages'], params['mag_blocks_per_stage'], params['mag_kernel_size'],
                                params['mag_downsampling_strides'])(mag_input)

    if SW:
        # Solar Wind Net
        sw_input = keras.layers.Input(shape=[params['sw_T0'], sw_data.shape[2]])
        if params['sw_type'] == 'residual':
            sw_net = _residual_1d_net(params['sw_fl_filters'], params['sw_fl_kernel_size'], params['sw_fl_strides'],
                                      params['sw_stages'], params['sw_blocks_per_stage'], params['sw_kernel_size'],
                                      params['sw_downsampling_strides'])(sw_input)
        elif params['sw_type'] == 'basic':
            sw_net = _basic_1d_net(params['sw_fl_filters'], params['sw_fl_kernel_size'], params['sw_fl_strides'],
                                   params['sw_stages'], params['sw_blocks_per_stage'], params['sw_kernel_size'],
                                   params['sw_downsampling_strides'])(sw_input)
        # Concatenate the two results, apply a dense layer
        last_layer = keras.layers.Concatenate()([sw_net, mag_net])
        inputs = [mag_input, sw_input]
        train_data = [mag_data[:, :, -params['mag_T0']:], sw_data[:, -params['sw_T0']:]]
        val_data = [mag_data_val[:, :, -params['mag_T0']:], sw_data_val[:, -params['sw_T0']:]]
    else:
        last_layer = mag_net
        inputs = mag_input
        train_data = mag_data[:, :, -params['mag_T0']:]
        val_data = mag_data_val[:, :, -params['mag_T0']:]

    if params['n_classes'] == 2:
        time_output = keras.layers.Dense(1, activation='sigmoid', name='time_output')(last_layer)
        losses = {'time_output': 'binary_crossentropy', 'strength_output': 'mse'}
        metrics = {'time_output': ['accuracy', utils.true_positive, utils.false_positive],
                   'strength_output': ['mse', 'mae']}
    else:
        time_output = keras.layers.Dense(params['n_classes'], activation='softmax', name='time_output')(last_layer)
        losses = {'time_output': 'sparse_categorical_crossentropy', 'strength_output': 'mse'}
        metrics = {'time_output': ['accuracy'], 'strength_output': ['mse', 'mae']}

    strength_output = keras.layers.Dense(1, name='strength_output')(last_layer)

    model = keras.models.Model(inputs=inputs, outputs=[time_output, strength_output])
    loss_weights = {'time_output': params['time_output_weight'], 'strength_output': 1}
    model.compile(optimizer='adam', loss=losses, loss_weights=loss_weights, metrics=metrics)

    hist = model.fit(train_data, y_train, batch_size=params['batch_size'], epochs=params['epochs'],
                     validation_data=(val_data, y_val), verbose=params['verbose'])

    return hist, model


def _basic_2d_net(fl_filters, fl_kernel_size, fl_strides, stages, blocks_per_stage, kernel_size, strides, flx2=True):

    def f(x):
        net = blocks.conv_batch_relu(filters=fl_filters, kernel_size=fl_kernel_size, strides=fl_strides)(x)

        filters = fl_filters
        if flx2:
            filters *= 2

        for stage in range(stages):
            for _ in range(blocks_per_stage - 1):
                net = blocks.conv_batch_relu(filters=filters, kernel_size=kernel_size, strides=[1, 1])(net)
            net = blocks.conv_batch_relu(filters=filters, kernel_size=kernel_size, strides=strides)(net)
            filters *= 2
        return keras.layers.GlobalAveragePooling2D()(net)

    return f


def _basic_1d_net(fl_filters, fl_kernel_size, fl_strides, stages, blocks_per_stage, kernel_size, strides, flx2=True):

    def f(x):
        net = blocks.conv_batch_relu_1d(filters=fl_filters, kernel_size=fl_kernel_size, strides=fl_strides)(x)

        filters = fl_filters
        if flx2:
            filters *= 2

        for stage in range(stages):
            for _ in range(blocks_per_stage - 1):
                net = blocks.conv_batch_relu_1d(filters=filters, kernel_size=kernel_size, strides=1)(net)
            net = blocks.conv_batch_relu_1d(filters=filters, kernel_size=kernel_size, strides=strides)(net)
            filters *= 2
        return keras.layers.GlobalAveragePooling1D()(net)

    return f


def _residual_2d_net(fl_filters, fl_kernel_size, fl_strides, stages, blocks_per_stage, kernel_size, strides):

    def f(x):
        net = keras.layers.Conv2D(filters=fl_filters, kernel_size=fl_kernel_size, strides=fl_strides, padding="same")(x)

        filters = fl_filters

        for stage in range(stages):
            filters *= 2
            net = blocks.res_block_2d(filters=filters, kernel_size=kernel_size, strides=strides)(net)
            for _ in range(blocks_per_stage - 1):
                net = blocks.res_block_2d(filters=filters, kernel_size=kernel_size, strides=[1, 1])(net)

        net = keras.layers.BatchNormalization()(net)
        net = keras.layers.ReLU()(net)
        return keras.layers.GlobalAveragePooling2D()(net)

    return f


def _residual_1d_net(fl_filters, fl_kernel_size, fl_strides, stages, blocks_per_stage, kernel_size, strides):

    def f(x):
        net = keras.layers.Conv1D(filters=fl_filters, kernel_size=fl_kernel_size, strides=fl_strides)(x)

        filters = fl_filters

        for stage in range(stages):
            filters *= 2
            net = blocks.res_block_1d(filters=filters, kernel_size=kernel_size, strides=strides)(net)
            for _ in range(blocks_per_stage - 1):
                net = blocks.res_block_1d(filters=filters, kernel_size=kernel_size, strides=1)(net)

        net = keras.layers.BatchNormalization()(net)
        net = keras.layers.ReLU()(net)
        return keras.layers.GlobalAveragePooling1D()(net)

    return f
