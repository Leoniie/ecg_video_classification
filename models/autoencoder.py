from datetime import datetime
from keras.models import Model
import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, Flatten, Conv1D, MaxPooling1D, Reshape, UpSampling2D
from keras.layers import Dense, Dropout, Input, Embedding
from keras.models import Sequential
from helpers.plotter import plot
from helpers.metrics import confusion_metric_vis
from keras import regularizers


def build_encoder(img_width, img_height):
    # define CNN model

    input_layer = Input(shape=(img_width, img_height, 1))
    # Encoder
    en1 = Conv2D(16, (10, 10), activation='relu', padding='same')(input_layer)
    max1 = MaxPooling2D(pool_size=(2, 2)) (en1)
    en2 = Conv2D(8, (10, 10), activation='relu', padding='same') (max1)
    max2 = MaxPooling2D(pool_size=(2, 2)) (en2)

    #Decoder

    de1 = Conv2D(8, (10, 10), activation='relu', padding='same') (max2)
    Up1 = UpSampling2D(size=(2,2)) (de1)
    de2 = Conv2D(16, (10, 10), activation='relu', padding='same') (Up1)
    Up2 = UpSampling2D(size=(2,2)) (de2)

    Out_dec = Conv2D(1, (3,3), activation='sigmoid', padding='same') (Up2)

    autoencoder = Model(input_layer, Out_dec)
    autoencoder.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    encode = Model(input_layer, max2)

    return autoencoder, encode


def evaluate_auto(X, X_train, X_test):
    # Hyperparameter!

    #filter = 32 disabled
    patience = 10
    batch_size = 32
    epochs = 5
    kernel_size = 5
    print("Shape before Model: ", X.shape)
    img_width, img_height = X.shape[1], X.shape[2]


    model, encode = build_encoder(img_width=img_width, img_height=img_height)  # , Neurons = Neurons
    # print('\nModel with input size {}, output size {}, {} conv filters of length {}'.format(model.input_shape))

    print(model.summary())


    earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0, patience=patience, verbose=2,
                              mode='auto')
    time_before = datetime.now()
    model.fit(X, X,
              epochs=epochs, batch_size=batch_size, validation_split=0.2, shuffle=True, callbacks=[earlystop])  # , class_weight=class_weights
    time_after = datetime.now()

    print("fitting took {} seconds".format(time_after - time_before))

    x_en = encode.predict(X_train)
    x_test_en = encode.predict(X_test)
    return x_en, x_test_en
