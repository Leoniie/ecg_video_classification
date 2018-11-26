# TODO: Create a simple sequential Conv2DLSTM NN
#Malte


def build_sequential():
    model = Sequential()
    # define CNN model
    model.add(TimeDistributed(Conv2D(...))
    model.add(TimeDistributed(MaxPooling2D(...)))
    model.add(TimeDistributed(Flatten()))
    # define LSTM model
    model.add(LSTM(...))
    model.add(Dense(...))
    # TODO: build function

def compile_sequential():
    pass
    # TODO: compile