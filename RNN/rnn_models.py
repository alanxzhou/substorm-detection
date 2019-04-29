import numpy as np
from keras.layers import GRU, LSTM, RNN, Input, Dense, concatenate
from keras.models import Sequential, Model
import utils

def train_functional_gru(X_train, y_train, X_val, y_val, params):

    if isinstance(X_train, list):
        SW = True
        mag_data, sw_data = X_train
        mag_data_val, sw_data_val = X_val

    _, n_steps_mag, n_features_mag = np.shape(mag_data)
    _, n_steps_sw, n_features_sw = np.shape(sw_data)

    # magnetic field input
    mag_input = Input(batch_shape=[params['batch_size'], n_steps_mag, n_features_mag])
    recurrent_mag = GRU(params['rnn_hidden_units'], return_sequences=True)(mag_input)
    for i in range(params['n_stacks']-1):
        recurrent_mag = GRU(params['rnn_hidden_units'], return_sequences=True)(recurrent_mag)

    # solar wind input
    sw_input = Input(batch_shape =[params['batch_size'], n_steps_sw, n_features_sw])
    recurrent_sw = GRU(params['rnn_hidden_units'], return_sequences=True)(mag_input)
    for i in range(params['n_stacks']-1):
        recurrent_sw = GRU(params['rnn_hidden_units'], return_sequences=True)(recurrent_sw)

    # recombining
    recurrent = concatenate([recurrent_mag, recurrent_sw])
    x = Dense(params['fc_hidden_size'], activation='relu')(recurrent)
    model_output = Dense(1, activation='sigmoid')(x)


    model = Model(inputs=[mag_input, sw_input],outputs=model_output)
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy', utils.true_positive, utils.false_positive])

    hist = model.fit(X_train[:, :, -params['T0']:], y_train, batch_size=params['batch_size'], epochs=params['epochs'],
                     validation_data=(X_val[:, :, -params['T0']:], y_val), verbose=params['verbose'])
    return hist, model


def train_basic_gru(X_train, y_train, X_val, y_val, params):
    """
    Trains basic sequential model with stacked GRU layers
    :param X_train:
    :param y_train:
    :param X_val:
    :param y_val:
    :param params:
    :return:
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