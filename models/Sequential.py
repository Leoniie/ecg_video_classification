# TODO: Create a simple sequential Conv2DLSTM NN
#Malte
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.losses import binary_crossentropy


def build_sequential(nb_steps=200, nb_width=100, nb_height=100):
    model = Sequential()
    # define CNN model
    model.add(TimeDistributed(Conv2D(16, x_size,activation='relu', padding = 'same',input_shape=(nb_steps, nb_width, nb_height) ))
    model.add(TimeDistributed(MaxPooling2D(2,2)))
    model.add(TimeDistributed(Flatten()))
    # define LSTM model
    model.add(LSTM(...))
    model.add(Dense(2,activation='softmax'))
    model.compile(optimizer='adamax',
              loss='binary_crossentropy',
              metrics=['accuracy'])
    # TODO: build function
    return model


def evaluate_sequential(X, y):
    # Hyperparameter!
    # hidden_layers=0  #not more than 2

    patience = 20

    epochs = 150

    # X = np.atleast_2d(X)
    # if X.shape[0] == 1:
    #    X = X.T

    nb_samples, nb_steps, nb_width, nb_height = X.shape
    print('\nfunctional_net ({} samples by {} series)'.format(nb_samples, nb_series))

    model = make_functional_LSTM(filter_length=filter_length,
                                 nb_input_series=nb_series, nb_input_channels=1,
                                 nb_filter=nb_filter)  # , Neurons = Neurons
    print('\nModel with input size {}, output size {}, {} conv filters of length {}'.format(model.input_shape,
                                                                                            model.output_shape,
                                                                                            nb_filter, filter_length))
    model.summary()

    print('\nInput features:', X.shape, '\nOutput labels:', y.shape, sep='\n')

    earlystop = keras.callbacks.EarlyStopping(monitor='val_f1', min_delta=0.0, patience=patience, verbose=2,
                                              mode='auto')
    time_before = datetime.now()
    model.fit([X, fft, bpm], y,
              epochs=epochs, batch_size=50, validation_split=0.2, shuffle=True, callbacks=[earlystop],
              class_weight=class_weights)  # , class_weight=class_weights
    time_after = datetime.now()

    print("fitting took {} seconds".format(time_after - time_before))
    y_pred = np.argmax(
        model.predict([preprocessing(df_test, resolution=0.75, resolution_type='resize'), fft_test, bpm_test]),
        axis=1)
    y_true = np.argmax(y, axis=1)
    try:
        confusion_metric_vis(y_true=y_true, y_pred=y_pred)
    except:
        pass

    y_test = np.argmax(
        model.predict([preprocessing(df_test, resolution=0.75, resolution_type='resize'), fft_test, bpm_test]),
        axis=1)

    return y_test