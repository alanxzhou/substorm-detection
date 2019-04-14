# univariate stacked lstm example
import numpy as np
import matplotlib.pyplot as plt
import glob
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.models import Sequential
from numpy import array
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder



# split a univariate sequence

params = {
    'mag_files': glob.glob("F:/data/mag_data_*.nc")[:2],
    'ss_file': "F:/data/substorms_2000_2018.csv",
    'data_interval': 96,
    'prediction_interval': 96,
    'val_size': 512,
    'batch_size': 32,
    'model_name': "Wider_Net"
}

def split_sequence(sequence, n_steps):
    n_points = np.shape(sequence)[1]
    X, y = [],[]
    for i in range(n_points):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > n_points-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[:,i:end_ix], sequence[0,end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

# define input sequence
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
triple_seq = np.vstack((raw_seq,raw_seq,raw_seq))
# choose a number of time steps
n_steps = 3
# split into samples
X, y = split_sequence(triple_seq, n_steps)

X = np.array([1,2,3,0,1,2,3,0,1,2,3,0])
X = np.reshape(X,(1,len(X)))
X, _ = split_sequence(X,n_steps)
n_samples = np.shape(X)[0]
y = np.array([1,0,0,0,1,0,0,0,1])
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(y)
onehot_encoder = OneHotEncoder(sparse = False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape(((n_samples,n_steps,n_features)))
#X = X.reshape((X.shape[0], X.shape[1], n_features))
# define model
yhat_all = []
for ii in range(10):

    print(ii)

    model = Sequential()
    #model.add(GRU(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
    #model.add(GRU(50, activation='relu'))
    model.add(GRU(50,  return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(GRU(50))
    model.add(Dense(50,activation='relu'))
    model.add(Dense(2, activation = 'softmax'))
    #model.add(Dense(1))
    #model.compile(optimizer='adam', loss='mse')
    model.compile(optimizer='adam', loss='binary_crossentropy')
    # fit model
    model.fit(X, onehot_encoded, epochs=200, verbose=0)
    # demonstrate prediction
    #temp = array([70, 80, 90])
    #x_input = np.vstack((temp,temp,temp))
    #x_input = np.array([[1,2,3],[2,3,1]])
    x_input = np.array([2,3,0])
    x_input = x_input.reshape((1, n_steps, n_features))
    yhat = model.predict_classes(x_input, verbose=0)
    #yhat_all.append(yhat)
    print('yhat:%s' %yhat)

    model = None

#print(sum(yhat_all))