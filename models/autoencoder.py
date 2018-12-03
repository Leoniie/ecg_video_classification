from datetime import datetime
from keras.models import Model
import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import Conv3D, MaxPooling3D, Flatten, Conv1D, MaxPooling1D, Reshape, UpSampling3D
from keras.layers import Dense, Dropout, Input, Embedding
from keras.models import Sequential
from helpers.plotter import plot
from helpers.metrics import confusion_metric_vis
from keras import regularizers


def build_encoder(nb_steps, nb_width, nb_height, input_channels, kernel_size):
    # define CNN model
    input_layer = Input(shape=(nb_steps, nb_width, nb_height, input_channels))
    # Encoder
    en1 = Conv3D(16, kernel_size, activation='relu', padding='same')(input_layer)
    max1 = MaxPooling3D(pool_size=(2, 2, 2)) (en1)
    en2 = Conv3D(8, kernel_size, activation='relu', padding='same') (max1)
    max2 = MaxPooling3D(pool_size=(2, 2, 2)) (en2)

    #Decoder

    de1 = Conv3D(8, kernel_size, activation='relu', padding='same') (max2)
    Up1 = UpSampling3D(size=(2,2,2)) (de1)
    de2 = Conv3D(16, kernel_size, activation='relu', padding='same') (Up1)
    Up2 = UpSampling3D(size=(2,2,2)) (de2)

    Out_dec = Conv3D(1, (3,3,3), strides=(1,1,1), activation='sigmoid', padding='same') (Up2)

    autoencoder = Model(input_layer, Out_dec)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
    encode = Model(input_layer, max2)

    return autoencoder, encode


def evaluate_auto(X):
    # Hyperparameter!

    #filter = 32 disabled
    patience = 10
    batch_size = 32
    epochs = 20
    kernel_size = 5
    print("Shape before Model: ", X.shape)
    nb_samples, nb_steps, nb_width, nb_height, input_channels = X.shape
    print('\nfunctional_net ({} samples by {} series)'.format(nb_samples, nb_steps))


    model, encode = build_encoder(kernel_size=kernel_size, nb_steps=nb_steps, nb_width=nb_width, nb_height=nb_height,
                          input_channels=input_channels)  # , Neurons = Neurons
    # print('\nModel with input size {}, output size {}, {} conv filters of length {}'.format(model.input_shape))

    print(model.summary())


    earlystop = EarlyStopping(monitor='val_accuracy', min_delta=0.0, patience=patience, verbose=2,
                              mode='auto')
    time_before = datetime.now()
    model.fit(X, X,
              epochs=epochs, batch_size=batch_size, validation_split=0.2, shuffle=True, callbacks=[earlystop])  # , class_weight=class_weights
    time_after = datetime.now()

    print("fitting took {} seconds".format(time_after - time_before))

    x_en = encode.predict(X)
    return x_en
