from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Input, BatchNormalization, \
    Activation, concatenate


# No modifications
def base_model(x_train, classes):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=x_train[0].shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.7))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    return model


# Please make sure to correct only this part.
def custom_model(x_train, classes):
    input_layer = Input(shape=x_train[0].shape)
    x = Conv2D(32, kernel_size=(3, 3), padding='same', name='conv_1')(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, kernel_size=(3, 3), padding='same', name='conv_2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    mix = concatenate([input_layer, x])

    mp = MaxPooling2D(2)(mix)

    x = Conv2D(32, kernel_size=(3, 3), name='conv_3')(mp)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, kernel_size=(3, 3), name='conv_4')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    y = Conv2D(32, kernel_size=(3, 3), name='conv_5')(mp)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv2D(64, kernel_size=(3, 3), name='conv_6')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    x = MaxPooling2D(2)(x)
    y = MaxPooling2D(2)(y)
    mix2 = concatenate([x, y])

    x = MaxPooling2D(2)(mix2)

    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(512)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256)(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(64)(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(classes)(x)
    outputs = Activation('softmax')(x)

    model = Model(inputs=input_layer, outputs=outputs)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    return model
