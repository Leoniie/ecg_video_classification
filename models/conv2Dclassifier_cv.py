
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras import regularizers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

def to2D(optimizer = 'rmsprop'):




    model = Sequential()
    model.add(Conv2D(32, (10, 10), input_shape=(96,96, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Conv2D(32, (3, 3), kernel_regularizer=regularizers.l2(0.2)) )
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Conv2D(64, (10, 10)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64,  kernel_regularizer=regularizers.l2(0.2)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])



    return model


def Wrapper(X,y,X_test):

    wrapped = KerasClassifier(build_fn=to2D, epochs=10, batch_size=16, verbose=1)
    # evaluate using 10-fold cross validation
    optimizer = ['rmsprop']

    param_grid = dict(optimizer=optimizer)
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    grid = GridSearchCV(estimator=wrapped, param_grid=param_grid, cv=kfold)
    grid.fit(X, y)
    y_test = grid.predict(X_test)


    return y_test

