import keras
from keras.layers import GRU, LSTM, RNN, Input, Dense, concatenate, Flatten
from keras.models import Sequential, Model
import numpy as np
import utils


def train_functional_rnn_combined(X_train, y_train, X_val, y_val, params):

    if isinstance(X_train, list):
        SW = True
        mag_data, sw_data = X_train
        mag_data_val, sw_data_val = X_val

    _, n_steps_mag, n_features_mag = np.shape(mag_data)
    _, n_steps_sw, n_features_sw = np.shape(sw_data)

    # magnetic field input
    mag_input = Input(batch_shape=[params['batch_size'], n_steps_mag, n_features_mag])

    # solar wind input
    sw_input = Input(batch_shape=[params['batch_size'], n_steps_sw, n_features_sw])

    if params['rnn_type'].upper() == 'GRU':
        recurrent_mag = GRU(params['rnn_hidden_units'], return_sequences=True)(mag_input)
        for i in range(params['n_stacks']-1):
            recurrent_mag = GRU(params['rnn_hidden_units'], return_sequences=True)(recurrent_mag)

        recurrent_sw = GRU(params['rnn_hidden_units'], return_sequences=True)(mag_input)
        for i in range(params['n_stacks'] - 1):
            recurrent_sw = GRU(params['rnn_hidden_units'], return_sequences=True)(recurrent_sw)

    elif params['rnn_type'].upper() == 'LSTM':
        recurrent_mag = LSTM(params['rnn_hidden_units'], return_sequences=True)(mag_input)
        for i in range(params['n_stacks']-1):
            recurrent_mag = LSTM(params['rnn_hidden_units'], return_sequences=True)(recurrent_mag)

        recurrent_sw = LSTM(params['rnn_hidden_units'], return_sequences=True)(mag_input)
        for i in range(params['n_stacks'] - 1):
            recurrent_sw = LSTM(params['rnn_hidden_units'], return_sequences=True)(recurrent_sw)

    elif params['rnn_type'].upper() == 'RNN':
        recurrent_mag = RNN(params['rnn_hidden_units'], return_sequences=True)(mag_input)
        for i in range(params['n_stacks']-1):
            recurrent_mag = RNN(params['rnn_hidden_units'], return_sequences=True)(recurrent_mag)

        recurrent_sw = RNN(params['rnn_hidden_units'], return_sequences=True)(mag_input)
        for i in range(params['n_stacks'] - 1):
            recurrent_sw = RNN(params['rnn_hidden_units'], return_sequences=True)(recurrent_sw)

    # recombining
    recurrent = concatenate([recurrent_mag, recurrent_sw])
    last_layer = Dense(params['fc_hidden_size'], activation='relu')(recurrent)
    last_layer = Flatten()(last_layer)

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
    print(model.summary())

    loss_weights = {'time_output': params['time_output_weight'], 'strength_output': 1}
    model.compile(optimizer='adam', loss=losses, loss_weights=loss_weights, metrics=metrics)

    hist = model.fit(X_train, y_train, batch_size=params['batch_size'], epochs=params['epochs'],
                     validation_data=(X_val, y_val), verbose=params['verbose'])

    return hist, model


def train_functional_rnn_separate(X_train, y_train, X_val, y_val, params):

    if isinstance(X_train, list):
        SW = True
        mag_data, sw_data = X_train
        mag_data_val, sw_data_val = X_val
        y_time, y_strength = y_train
        y_time_val, y_strength_val = y_val

    _, n_steps_mag, n_features_mag = np.shape(mag_data)
    _, n_steps_sw, n_features_sw = np.shape(sw_data)

    # magnetic field input
    mag_input = Input(batch_shape=[params['batch_size'], n_steps_mag, n_features_mag])

    # solar wind input
    sw_input = Input(batch_shape=[params['batch_size'], n_steps_sw, n_features_sw])

    if params['rnn_type'].upper() == 'GRU':
        recurrent_mag = GRU(params['rnn_hidden_units'], return_sequences=True)(mag_input)
        for i in range(params['n_stacks']-1):
            recurrent_mag = GRU(params['rnn_hidden_units'], return_sequences=True)(recurrent_mag)

        recurrent_sw = GRU(params['rnn_hidden_units'], return_sequences=True)(mag_input)
        for i in range(params['n_stacks'] - 1):
            recurrent_sw = GRU(params['rnn_hidden_units'], return_sequences=True)(recurrent_sw)

    elif params['rnn_type'].upper() == 'LSTM':
        recurrent_mag = LSTM(params['rnn_hidden_units'], return_sequences=True)(mag_input)
        for i in range(params['n_stacks']-1):
            recurrent_mag = LSTM(params['rnn_hidden_units'], return_sequences=True)(recurrent_mag)

        recurrent_sw = LSTM(params['rnn_hidden_units'], return_sequences=True)(mag_input)
        for i in range(params['n_stacks'] - 1):
            recurrent_sw = LSTM(params['rnn_hidden_units'], return_sequences=True)(recurrent_sw)

    elif params['rnn_type'].upper() == 'RNN':
        recurrent_mag = RNN(params['rnn_hidden_units'], return_sequences=True)(mag_input)
        for i in range(params['n_stacks']-1):
            recurrent_mag = RNN(params['rnn_hidden_units'], return_sequences=True)(recurrent_mag)

        recurrent_sw = RNN(params['rnn_hidden_units'], return_sequences=True)(mag_input)
        for i in range(params['n_stacks'] - 1):
            recurrent_sw = RNN(params['rnn_hidden_units'], return_sequences=True)(recurrent_sw)

    # recombining
    recurrent = concatenate([recurrent_mag, recurrent_sw])
    last_layer = Dense(params['fc_hidden_size'], activation='relu')(recurrent)
    last_layer = Flatten()(last_layer)

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

    model_time = Model(inputs=[mag_input, sw_input], outputs=time_output)
    model_time.compile(optimizer='adam', loss=losses['time_output'], metrics=metrics['time_output'])
    hist_time = model_time.fit(X_train, y_time, batch_size=params['batch_size'], epochs=params['epochs'],
                     validation_data=(X_val, y_time_val), verbose=params['verbose'])

    model_strength = Model(inputs=[mag_input, sw_input], outputs=strength_output)
    model_strength.compile(optimizer='adam',loss=losses['strength_output'], metrics=metrics['strength_output'])
    hist_strength = model_strength.fit(X_train, y_strength, batch_size=params['batch_size'], epochs=params['epochs'],
                     validation_data=(X_val, y_strength_val), verbose=params['verbose'])

    if params['verbose']:
        print(model_time.summary())
        print(model_strength.summary())

    model = [model_time, model_strength]
    hist = [hist_time, hist_strength]

    return hist, model


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
        model.add(GRU(params['rnn_hidden_units'], return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(GRU(params['rnn_hidden_units']))
    model.add(Dense(params['fc_hidden_size'], activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy', utils.true_positive, utils.false_positive])
    hist = model.fit(X_train, y_train, batch_size=params['batch_size'], epochs=params['epochs'],
                     validation_data=(X_val, y_val))

    return hist, model


def train_basic_lstm(X_train, y_train, X_val, y_val, params):
    model = Sequential()
    _, n_steps, n_features = np.shape(X_train)
    for i in range(params['n_stacks']-1):
        model.add(LSTM(params['rnn_hidden_units'], return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(LSTM(params['rnn_hidden_units']))
    model.add(Dense(params['fc_hidden_size'], activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy', utils.true_positive, utils.false_positive])
    hist = model.fit(X_train, y_train, batch_size=params['batch_size'], epochs=params['epochs'],
                     validation_data=(X_val, y_val))

    return hist, model


def train_basic_rnn(X_train, y_train, X_val, y_val, params):
    model = Sequential()
    _, n_steps, n_features = np.shape(X_train)
    for i in range(params['n_stacks']-1):
        model.add(RNN(params['rnn_hidden_units'], return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(RNN(params['rnn_hidden_units']))
    model.add(Dense(params['fc_hidden_size'], activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy', utils.true_positive, utils.false_positive])
    hist = model.fit(X_train, y_train, batch_size=params['batch_size'], epochs=params['epochs'],
                     validation_data=(X_val, y_val))

    return hist, model