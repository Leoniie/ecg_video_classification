# TODO: Create a simple sequential Conv2DLSTM NN
#Malte
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import TimeDistributed


def build_sequential(x_lengt, x_size= (100,100)):
    model = Sequential()
    # define CNN model
    model.add(TimeDistributed(Conv2D(16, x_size,activation='relu', padding = 'same',input_shape=(100,100, x_length) ))
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

def compile_sequential():
    pass
    # TODO: compile