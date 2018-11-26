# TODO: fix the error, maybe it's something with the input shape?
#Malte
from __future__ import print_function, division
from datetime import datetime
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.losses import binary_crossentropy
from helpers.metrics import final_metric
from helpers.preprocessing import preprocessing_scaled
from helpers.preprocessing import preprocessing
from keras.layers import TimeDistributed
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import Conv2D

from keras.models import Sequential
from keras.layers import  Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.layers import TimeDistributed
from keras.layers import LSTM

from keras.layers import Dense

def build_sequential(nb_steps, nb_width, nb_height, nb_channels):
    # define CNN model
    cnn = Sequential()
    cnn.add(Conv2D(nb_channels,2,activation='relu', padding = 'same',input_shape=(nb_steps,nb_width, nb_height)))
    cnn.add(MaxPooling2D(pool_size=(2,2)))
    cnn.add(Flatten())
    model = Sequential()

    #TODO: what is x_size?


    # define LSTM model
    model.add(TimeDistributed(cnn))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(2,activation='softmax'))
    model.compile(optimizer='adamax',
              loss='binary_crossentropy',
              metrics=['accuracy',final_metric])

    return model


def evaluate_sequential(X, y):
    # Hyperparameter!
    # hidden_layers=0  #not more than 2
    nb_channels=16
    patience = 3
    batch_size=1
    epochs = 150

    # X = np.atleast_2d(X)
    # if X.shape[0] == 1:
    #    X = X.T

    nb_samples, nb_steps, nb_width, nb_height = X.shape
    print('\nfunctional_net ({} samples by {} series)'.format(nb_samples, nb_steps))

    model = build_sequential(nb_steps, nb_width, nb_height, nb_channels)  # , Neurons = Neurons
    #print('\nModel with input size {}, output size {}, {} conv filters of length {}'.format(model.input_shape))


    print('\nInput features:', X.shape, '\nOutput labels:', y.shape, sep='\n')

    earlystop = EarlyStopping(monitor='val_final_metric', min_delta=0.0, patience=patience, verbose=2,
                                              mode='auto')
    time_before = datetime.now()
    model.fit(X, y,
              epochs=epochs, batch_size=batch_size, validation_split=0.2, shuffle=True, callbacks=[earlystop],
              )  # , class_weight=class_weights
    time_after = datetime.now()

    print("fitting took {} seconds".format(time_after - time_before))
    y_pred = np.argmax(
        model.predict(preprocessing(x_test)),
        axis=1)
    y_true = np.argmax(y, axis=1)
    try:
        confusion_metric_vis(y_true=y_true, y_pred=y_pred)
    except:
        pass

    y_test = np.argmax(
        model.predict(preprocessing(x_test)),
        axis=1)

    return y_test