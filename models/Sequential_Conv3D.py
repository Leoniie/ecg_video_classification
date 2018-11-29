from datetime import datetime

import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import Conv3D, MaxPooling3D, Flatten, Cropping3D
from keras.layers import Dense, Dropout
from keras.models import Sequential

from helpers.metrics import confusion_metric_vis


def build_sequential(nb_steps, nb_width, nb_height, input_channels, filter, kernel_size):
    # define CNN model
    model = Sequential()
    model.add(Cropping3D(1, data_format="channels_last", input_shape=(nb_steps, nb_width, nb_height, input_channels)))
    model.add(Conv3D(filter, kernel_size, activation='relu', data_format='channels_last'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Conv3D(filter, kernel_size, activation='relu', padding='same', data_format='channels_last'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Conv3D(filter, kernel_size, activation='relu', padding='same', data_format='channels_last'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Conv3D(filter, kernel_size, activation='relu', padding='same', data_format='channels_last'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Flatten())
    model.add(Dense(20, activation='relu', name='first_dense'))
    model.add(Dropout(0.5))
    # model.add(Dense(32, activation="relu", name="second_dense"))
    # model.add(Dropout(0.4))
    model.add(Dense(1, activation='softmax', name="last_dense"))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])  # ,final_metric

    return model


def evaluate_sequential(X, y, x_test):
    # Hyperparameter!

    filter = 32
    patience = 5
    batch_size = 1
    epochs = 40
    kernel_size = 3

    # X = np.atleast_2d(X)
    # if X.shape[0] == 1:
    #    X = X.T

    nb_samples, nb_steps, nb_width, nb_height, input_channels = X.shape
    print('\nfunctional_net ({} samples by {} series)'.format(nb_samples, nb_steps))

    model = build_sequential(kernel_size=kernel_size, nb_steps=nb_steps, nb_width=nb_width, nb_height=nb_height,
                             filter=filter, input_channels=input_channels)  # , Neurons = Neurons
    # print('\nModel with input size {}, output size {}, {} conv filters of length {}'.format(model.input_shape))

    print(model.summary())
    print('\nInput features:', X.shape, '\nOutput labels:', y.shape, sep='\n')

    earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0, patience=patience, verbose=2,
                              mode='auto')
    time_before = datetime.now()
    model.fit(X, y,
              epochs=epochs, batch_size=batch_size, validation_split=0.2, shuffle=True, callbacks=[earlystop],
              )  # , class_weight=class_weights
    time_after = datetime.now()

    print("fitting took {} seconds".format(time_after - time_before))
    # y_pred = np.argmax(model.predict(x_test), axis=1)
    # y_true = np.argmax(y, axis=1)

    y_pred = model.predict(x_test)
    y_true = y
    try:
        confusion_metric_vis(y_true=y_true, y_pred=y_pred)
    except:
        pass

    y_test = np.argmax(model.predict(x_test), axis=1)

    return y_test
