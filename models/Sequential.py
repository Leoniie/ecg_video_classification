# TODO: fix the error, maybe it's something with the input shape?
#Malte
from keras.layers import Dense, Dropout, Activation, LSTM
from keras.layers import Conv2D, MaxPooling2D, Flatten, Reshape
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import EarlyStopping
from datetime import datetime
from keras.layers.pooling import GlobalAveragePooling1D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import Model
from helpers.metrics import final_metric, confusion_metric_vis

def build_sequential(nb_steps, nb_width, nb_height, nb_channels, input_channels, kernel_size):
    # define CNN model
    model = Sequential()
    model.add(TimeDistributed(Conv2D(nb_channels, kernel_size, activation='relu'), input_shape=(nb_steps, nb_width, nb_height, input_channels)))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    model.add(TimeDistributed(Conv2D(64, (4, 4), activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
    model.add(TimeDistributed(Dropout(0.5)))

    model.add(TimeDistributed(Flatten()))

    model.add(LSTM(20, return_sequences=False, name="lstm_layer"))
    #More Dense??
    model.add(Dense(2,activation='softmax',name="second_dense"))
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy']) #,final_metric

    return model


def evaluate_sequential(X, y):
    # Hyperparameter!
    # hidden_layers=0  #not more than 2
    nb_channels=16
    patience = 3
    batch_size=1
    epochs = 20
    kernel_size = 2

    # X = np.atleast_2d(X)
    # if X.shape[0] == 1:
    #    X = X.T

    nb_samples, nb_steps, nb_width, nb_height, input_channels = X.shape
    print('\nfunctional_net ({} samples by {} series)'.format(nb_samples, nb_steps))

    model = build_sequential(kernel_size=kernel_size, nb_steps=nb_steps, nb_width=nb_width, nb_height=nb_height, nb_channels=nb_channels, input_channels=input_channels)  # , Neurons = Neurons
    #print('\nModel with input size {}, output size {}, {} conv filters of length {}'.format(model.input_shape))


    print('\nInput features:', X.shape, '\nOutput labels:', y.shape, sep='\n')

    earlystop = EarlyStopping(monitor='val_accuracy', min_delta=0.0, patience=patience, verbose=2,
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