import numpy as np
from keras.layers import GRU, LSTM, RNN, Input, Dense
from keras.models import Sequential Model
import utils

def train_functional_gru(X_train, y_train, X_val, y_val, params):
    _, n_steps, n_features = np.shape(X_train)
    main_input = Input(batch_shape=[params['batch_size'], n_steps, n_features])
    recurrent = GRU(params['rnn_hidden_units'], return_sequences=True)(main_input)
    for i in range(params['n_stacks']-1):
        recurrent = GRU(params['rnn_hidden_units'], return_sequences=True)(recurrent)
    x = Dense(params['fc_hidden_size'], activation='relu')(recurrent)
    model_output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=main_input,outputs=model_output)
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