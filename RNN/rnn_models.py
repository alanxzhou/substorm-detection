import keras
from keras.layers import GRU, LSTM, SimpleRNN, Input, Dense, concatenate, Flatten, Conv1D, MaxPooling1D
from keras.models import Sequential, Model
import numpy as np
import utils


def train_functional_rnn_combined(X_train, y_train, X_val, y_val, params):

    mag_data, sw_data = X_train
    mag_data_val, sw_data_val = X_val

    _, n_steps_mag, n_features_mag = np.shape(mag_data)
    _, n_steps_sw, n_features_sw = np.shape(sw_data)

    # magnetic field input
    mag_input = Input(batch_shape=[None, n_steps_mag, n_features_mag])

    # solar wind input
    sw_input = Input(batch_shape=[None, n_steps_sw, n_features_sw])

    recurrent_mag, recurrent_sw = stacked_rnn_layer(params, mag_input, sw_input)

    # recombining
    recurrent = concatenate([recurrent_mag, recurrent_sw])
    last_layer = Dense(params['fc_hidden_size'], activation='relu')(recurrent)

    if params['n_classes'] == 2:
        time_output = keras.layers.Dense(2, activation='sigmoid', name='time_output')(last_layer)
        losses = {'time_output': 'binary_crossentropy', 'strength_output': 'mse'}
        metrics = {'time_output': ['accuracy', utils.true_positive, utils.false_positive],
                   'strength_output': ['mse', 'mae']}
    else:
        time_output = keras.layers.Dense(params['n_classes'], activation='softmax', name='time_output')(last_layer)
        losses = {'time_output': 'sparse_categorical_crossentropy', 'strength_output': 'mse'}
        metrics = {'time_output': ['accuracy'], 'strength_output': ['mse', 'mae']}

    strength_output = keras.layers.Dense(1, name='strength_output')(last_layer)
    model = Model(inputs=[mag_input, sw_input], outputs=[time_output, strength_output])
    if params['verbose']:
        print(model.summary())

    loss_weights = {'time_output': params['time_output_weight'], 'strength_output': 1}
    model.compile(optimizer='adam', loss=losses, loss_weights=loss_weights, metrics=metrics)

    hist = model.fit(X_train, y_train, batch_size=params['batch_size'], epochs=params['epochs'],
                     validation_data=(X_val, y_val), verbose=params['verbose'])

    return hist, model


def train_functional_rnn_single(X_train, y_train, X_val, y_val, params):

    mag_data, sw_data = X_train
    y_time, y_strength = y_train
    y_time_val, y_strength_val = y_val

    _, n_steps_mag, n_features_mag = np.shape(mag_data)
    _, n_steps_sw, n_features_sw = np.shape(sw_data)

    # magnetic field input
    mag_input = Input(batch_shape=[None, n_steps_mag, n_features_mag])

    # solar wind input
    sw_input = Input(batch_shape=[None, n_steps_sw, n_features_sw])
    #filtered_sw = Conv1D(filters=1, kernel_size=1, data_format='channels_last')(sw_input)

    recurrent_mag, recurrent_sw = stacked_rnn_layer(params, mag_input, sw_input)

    # recombining
    recurrent = concatenate([recurrent_mag, recurrent_sw])
    last_layer = Dense(params['fc_hidden_size'], activation='relu')(recurrent)

    if params['output_type'].lower() == 'time':
        if params['n_classes'] == 2:
            time_output = keras.layers.Dense(2, activation='sigmoid', name='time_output')(last_layer)
            loss = 'binary_crossentropy'
            metrics = ['accuracy', utils.true_positive, utils.false_positive]
        else:
            time_output = keras.layers.Dense(params['n_classes'], activation='softmax', name='time_output')(last_layer)
            loss = 'sparse_categorical_cross_entropy'
            metrics = ['accuracy']
        model = Model(inputs=[mag_input, sw_input], outputs=time_output)
        y_train, y_val = y_time, y_time_val
    elif params['output_type'].lower() == 'strength':
        strength_output = keras.layers.Dense(1, name='strength_output')(last_layer)
        model = Model(inputs=[mag_input, sw_input], outputs=strength_output)
        loss = 'mse'
        metrics = ['mse', 'mae']
        y_train, y_val = y_strength, y_strength_val
    else:
        print('Not a valid output type')
        return

    model.compile(optimizer='adam', loss=loss, metrics=metrics)

    if params['verbose']:
        print(model.summary())

    hist = model.fit(X_train, y_train, batch_size=params['batch_size'], epochs=params['epochs'],
                     validation_data=(X_val, y_val), verbose=params['verbose'])

    return hist, model


def stacked_rnn_layer(params, mag_input, sw_input, sw_downsample=2):

    if params['rnn_type'].upper() == 'GRU':
        recurrent_mag = mag_input
        for i in range(params['n_stacks'] - 1):
            recurrent_mag = GRU(params['rnn_hidden_units'], return_sequences=True, dropout=params['dropout_rate'])(recurrent_mag)
        recurrent_mag = GRU(params['rnn_hidden_units'], dropout=params['dropout_rate'])(recurrent_mag)

        recurrent_sw = sw_input
        for i in range(params['n_stacks'] - 1):
            recurrent_sw = GRU(params['rnn_hidden_units'], return_sequences=True, dropout=params['dropout_rate'])(recurrent_sw)
        recurrent_sw = GRU(params['rnn_hidden_units'], dropout=params['dropout_rate'])(recurrent_sw)

    elif params['rnn_type'].upper() == 'LSTM':
        recurrent_mag = mag_input
        for i in range(params['n_stacks']-1):
            recurrent_mag = LSTM(params['rnn_hidden_units'], return_sequences=True, dropout=['dropout_rate'])(recurrent_mag)
        recurrent_mag = LSTM(params['rnn_hidden_units'], return_sequences=True, dropout=params['dropout_rate'])(recurrent_mag)

        recurrent_sw = sw_input
        for i in range(params['n_stacks'] - 1):
            recurrent_sw = LSTM(params['rnn_hidden_units'], return_sequences=True, dropout=['dropout_rate'])(recurrent_sw)
        recurrent_sw = LSTM(params['rnn_hidden_units'], dropout=params['dropout_rate'])(recurrent_sw)

    elif params['rnn_type'].upper() == 'RNN':
        recurrent_mag = SimpleRNN(params['rnn_hidden_units'], return_sequences=True)(mag_input)
        for i in range(params['n_stacks']-1):
            recurrent_mag = SimpleRNN(params['rnn_hidden_units'], return_sequences=True)(recurrent_mag)

        recurrent_sw = SimpleRNN(params['rnn_hidden_units'], return_sequences=True)(sw_input)
        for i in range(params['n_stacks'] - 1):
            recurrent_sw = SimpleRNN(params['rnn_hidden_units'], return_sequences=True)(recurrent_sw)

    #if sw_downsample > 1:
    #    recurrent_sw = MaxPooling1D(pool_size=sw_downsample, strides=sw_downsample, data_format='channels_last')(recurrent_sw)

    #recurrent_mag, recurrent_sw = Flatten()(recurrent_mag), Flatten()(recurrent_sw)

    return recurrent_mag, recurrent_sw


def train_basic_gru(X_train, y_train, X_val, y_val, params):
    """
    Trains basic sequential model with stacked GRU layers
    :param params = {'batch_size',
                    'epochs',
                    'rnn_hidden_units',
                    'fc_hidden_size',
                    'n_stacks'}
    :return:
        hist: dict; history of training
        model: keras model object; trained model
    """
    model = Sequential()
    _, n_steps, n_features = np.shape(X_train)
    for i in range(params['n_stacks']-1):
        model.add(GRU(params['rnn_hidden_units'], return_sequences=True, dropout=params['dropout_rate'], input_shape=(n_steps, n_features)))
    model.add(GRU(params['rnn_hidden_units'], dropout=params['dropout_rate']))
    model.add(Dense(params['fc_hidden_size'], activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy', utils.true_positive, utils.false_positive])
    hist = model.fit(X_train, y_train, batch_size=params['batch_size'], epochs=params['epochs'],
                     validation_data=(X_val, y_val))

    return hist, model
