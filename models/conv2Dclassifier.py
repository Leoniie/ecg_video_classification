
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras import regularizers


def to2D(x_train,y_train,x_test):

    # dimensions of our images.
    img_width, img_height = x_train.shape[1],x_train.shape[2]

    nb_train_samples = x_train.shape[0]
    nb_validation_samples = x_test.shape[0]
    epochs = 10
    batch_size = 16



    input_shape = (img_width, img_height, 1)

    model = Sequential()
    model.add(Conv2D(32, (10, 10), input_shape=input_shape, name='conv2d1'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Conv2D(32, (3, 3), kernel_regularizer=regularizers.l2(0.01), name='conv2d2'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Conv2D(64, (10, 10), name='conv2d3'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64,  kernel_regularizer=regularizers.l2(0.01)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))






    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    print(model.summary())
    model.fit(x_train, y_train,epochs=epochs, batch_size=batch_size, validation_split=0.2, shuffle=True)


    y_test = model.predict(x_test)

    return y_test, model

