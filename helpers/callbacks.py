from keras.callbacks import EarlyStopping, TensorBoard
import os


def stopxper(patience: int, monitor: object) -> object:
    stop = EarlyStopping(monitor=monitor, min_delta=0, patience=patience,
                         verbose=2, mode='auto')

    return stop


def tensorboard():
    # TODO: configurate tensorboard
    log_dir = os.path.relpath('logs')
    board = TensorBoard(log_dir=log_dir, histogram_freq=0, batch_size=32, write_graph=True, write_grads=False,
                        write_images=False, update_freq='epoch')
    return board
