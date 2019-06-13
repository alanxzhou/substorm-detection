import keras
from detection import utils
from detection.CNN.models import _basic_1d_net, _basic_2d_net, _residual_1d_net, _residual_2d_net
import numpy as np

def train_residual_cnn(X_train, y_train, X_val, y_val, params):
    """ X_train: [Mag_data, SW_data]
    This network uses solar wind data. Basically, it's two networks, one for mag data, one for solar wind, both ending
    with global average pooling. The outputs from both networks are concatenated and passed through a softmax
    classifier.

    params = {'batch_size': batch_size, 'epochs': epochs, 'verbose': 2, 'n_classes': 1,

              'mag_T0': mag_T0, 'mag_stages': 3, 'mag_blocks_per_stage': 3, 'mag_downsampling_strides': [2, 2],
              'mag_flx2': False, 'mag_kernel_size': [3, 5], 'mag_fl_filters': 128, 'mag_fl_strides': [2, 3],
              'mag_fl_kernel_size': [2, 13],

              'Tw': Tw, 'sw_stages': 3, 'sw_blocks_per_stage': 1, 'sw_downsampling_strides': 2,
              'sw_flx2': True, 'sw_kernel_size': 9, 'sw_fl_filters': 32, 'sw_fl_strides': 2,
              'sw_fl_kernel_size': 13}
    """

    mag_data, sw_data = X_train
    mag_data_val, sw_data_val = X_val

    mag_input = keras.layers.Input(shape=[mag_data.shape[1], params['Tm'], mag_data.shape[-1]])
    if params['mag_type'] == 'residual':
        mag_net = _residual_2d_net(params['mag_fl_filters'], params['mag_fl_kernel_size'], params['mag_fl_strides'],
                                   params['mag_stages'], params['mag_blocks_per_stage'], params['mag_kernel_size'],
                                   params['mag_downsampling_strides'])(mag_input)
    elif params['mag_type'] == 'basic':
        mag_net = _basic_2d_net(params['mag_fl_filters'], params['mag_fl_kernel_size'], params['mag_fl_strides'],
                                params['mag_stages'], params['mag_blocks_per_stage'], params['mag_kernel_size'],
                                params['mag_downsampling_strides'])(mag_input)
    else:
        raise Exception('mag_type must be either \'basic\' or \'residual\'')

    if params['SW']:
        # Solar Wind Net
        sw_input = keras.layers.Input(shape=[params['Tw'], sw_data.shape[2]])
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
        train_data = [mag_data[:, :, -params['Tm']:], sw_data[:, -params['Tw']:]]
        val_data = [mag_data_val[:, :, -params['Tm']:], sw_data_val[:, -params['Tw']:]]
    else:
        last_layer = mag_net
        inputs = mag_input
        train_data = mag_data[:, :, -params['Tm']:]
        val_data = mag_data_val[:, :, -params['Tm']:]

    time_output = keras.layers.Dense(1, name='time_output')(last_layer)
    loss = {'time_output': 'mse'}
    metrics = {'time_output': ['mse', 'mae']}
    model = keras.models.Model(inputs=inputs, outputs=time_output)
    model.compile(optimizer='adam', loss=loss, metrics=metrics)

    hist = model.fit(train_data, y_train, batch_size=params['batch_size'], epochs=params['epochs'],
                     validation_data=(val_data, y_val), verbose=params['verbose'])

    return hist, model


if __name__ == "__main__":
    data_fn = "../data/TEST_DATASET.npz"
    data = np.load(data_fn)

    X_train = [data['mag_data_train'], data['sw_data_train']]
    y_train = data['y_train']
    X_val = [data['mag_data_test'], data['sw_data_test']]
    y_val = data['y_test']

    batch_size = 16
    epochs = 1
    mag_T0 = 64
    sw_T0 = 240
    params = {'batch_size': batch_size, 'epochs': epochs, 'verbose': 2,
              'time_output_weight': 1000000, 'SW': True,

              'Tm': 96, 'mag_stages': 1, 'mag_blocks_per_stage': 4,
              'mag_downsampling_strides': (2, 3),
              'mag_kernel_size': (2, 11), 'mag_fl_filters': 16,
              'mag_fl_strides': (1, 3),
              'mag_fl_kernel_size': (1, 13), 'mag_type': 'basic',

              'Tw': 192, 'sw_stages': 1, 'sw_blocks_per_stage': 1,
              'sw_downsampling_strides': 4, 'sw_kernel_size': 7, 'sw_fl_filters': 16,
              'sw_fl_strides': 3, 'sw_fl_kernel_size': 15, 'sw_type': 'residual'}
    hist, mod = train_residual_cnn(X_train, y_train, X_val, y_val, params)