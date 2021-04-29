# import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical


def data_processing(x_train, x_test, y_train, y_test, CLASSNUM):
    x_train = np.array(x_train) / 255
    x_test = np.array(x_test) / 255

    y_train = to_categorical(y_train, CLASSNUM)
    y_test = to_categorical(y_test, CLASSNUM)

    return x_train, x_test, y_train, y_test


def create_model(CLASSNUM, SHAPE):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                     kernel_initializer='he_uniform',
                     input_shape=SHAPE))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.7))
    model.add(Flatten())
    model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.1))
    model.add(Dense(CLASSNUM, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model