import numpy as np
from keras.layers import Dense
from keras.layers import GRU
from keras.models import Sequential

import utils

def train_basic_gru(X_train, y_train, X_val, y_val, params):
    model = Sequential()
    _, n_steps, n_features = np.shape(X_train)
    model.add(GRU(50, return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(GRU(50))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy', utils.true_positive, utils.false_positive])
    hist = model.fit(X_train, y_train, batch_size=params['batch_size'], epochs=params['epochs'],
                     validation_data=(X_val, y_val))

    return hist,model
