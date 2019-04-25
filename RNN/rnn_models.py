import numpy as np
from keras.layers import Dense
from keras.layers import GRU, LSTM, RNN
from keras.models import Sequential
import utils


def train_basic_gru(X_train, y_train, X_val, y_val, params):
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

    return hist,model


def train_basic_lstm(X_train, y_train, X_val, y_val, params):
    model = Sequential()
    _, n_steps, n_features = np.shape(X_train)
    model.add(LSTM(50, return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(LSTM(50))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy', utils.true_positive, utils.false_positive])
    hist = model.fit(X_train, y_train, batch_size=params['batch_size'], epochs=params['epochs'],
                     validation_data=(X_val, y_val))

    return hist,model


def train_basic_rnn(X_train, y_train, X_val, y_val, params):
    model = Sequential()
    _, n_steps, n_features = np.shape(X_train)
    model.add(RNN(50, return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(RNN(50))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy', utils.true_positive, utils.false_positive])
    hist = model.fit(X_train, y_train, batch_size=params['batch_size'], epochs=params['epochs'],
                     validation_data=(X_val, y_val))

    return hist,model
