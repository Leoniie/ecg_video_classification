from keras.callbacks import EarlyStopping, TensorBoard


def stopper(patience: int, monitor: object) -> object:
    stop = EarlyStopping(monitor=monitor, min_delta=0, patience=patience,
                        verbose=2, mode='auto')

    return stop


def tensorboard():

    # TODO: define the storage of the log file

    # TODO: gitignore log files

    # TODO: configurate tensorboard

    board = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False,
                        write_images=False, update_freq='epoch')
    return board