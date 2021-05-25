from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation, Flatten, Conv2D, MaxPooling2D, Input, concatenate
from tensorflow.keras.models import Model
import tensorflow as tf

class Inception(Model):
    def __init__(self, filter_size, kernel, strides, padding='same'):
        super(Inception, self).__init__()
        self.conv = Conv2D(filter_size, kernel_size=kernel, strides=strides, padding=padding)
        self.batch = BatchNormalization()
        self.act = Activation('relu')

    def __call__(self, x):
        x = self.conv(x)
        x = self.batch(x)
        x = self.act(x)

        return x


class Stem(Model):
    def __init__(self):
        super(Stem, self).__init__()
        self.conv1 = Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu')
        self.conv2 = Conv2D(32, (3, 3), padding='same', activation='relu')
        self.conv3 = Conv2D(64, (3, 3), activation='relu')
        self.max1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")
        self.conv4 = Conv2D(96, (3, 3), strides=(2, 2), activation='relu', padding='same')

        self.conv5 = Conv2D(64, (1, 1), activation='relu')
        self.conv6 = Conv2D(64, (7, 1), padding='same', activation='relu')
        self.conv7 = Conv2D(64, (1, 7), padding='same', activation='relu')
        self.conv8 = Conv2D(96, (3, 3), padding='same', activation='relu')

        self.conv9 = Conv2D(64, (1, 1), activation='relu')
        self.conv10 = Conv2D(96, (3, 3), padding='same', activation='relu')

        self.conv11 = Conv2D(192, (3, 3), strides=(2, 2), padding='same', activation='relu')
        self.max2 = MaxPooling2D(strides=(2, 2), padding="same")

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        z = self.max1(x)
        y = self.conv4(x)

        con = concatenate([z, y])

        x = self.conv5(con)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)

        y = self.conv9(con)
        y = self.conv10(y)

        con = concatenate([x, y])

        print(con.shape)
        y = self.conv11(con)
        z = self.max2(con)
        print(y.shape)
        print(z.shape)

        output = concatenate([y, z])

        return output


class BatchConvolution(Model):
    def __init__(self, filter_size, kernel_size, activation, kernel_initializer):
        super(BatchConvolution, self).__init__()
        self.conv = Conv2D(filter_size, kernel_size=kernel_size, kernel_initializer=kernel_initializer)
        self.batch = BatchNormalization()
        self.active = Activation(activation)

    def __call__(self, x):
        x = self.conv(x)
        x = self.batch(x)
        x = self.active(x)
        return x


def create_model(train_image, classnum):
    # # create model
    # input_layer = Input(shape=train_image[0].shape)
    # x = BatchConvolution(16, (3, 3), 'relu', 'he_normal')(input_layer)
    # x = BatchConvolution(32, (3, 3), 'relu', 'he_normal')(x)
    # x = MaxPooling2D()(x)
    # x = BatchConvolution(64, (3, 3), 'relu', 'he_normal')(x)
    # x = BatchConvolution(128, (3, 3), 'relu', 'he_normal')(x)
    # x = MaxPooling2D()(x)
    # x = Flatten()(x)
    # x = Dense(128, activation='relu', kernel_initializer='he_normal')(x)
    # x = Dense(64, activation='relu', kernel_initializer='he_normal')(x)
    # x = Dropout(0.5)(x)
    # x = Dense(32, activation='relu', kernel_initializer='he_normal')(x)
    # x = Dense(16, activation='relu', kernel_initializer='he_normal')(x)
    # x = Dropout(0.3)(x)

    # create model
    input_layer = Input(shape=train_image[0].shape)
    stem = Stem()(input_layer)
    filter = [32, 64]
    stride = [(1, 1), (2, 2)]

    x = Inception(filter_size=filter[0], kernel=(3, 3), strides=stride[0])(stem)
    x = Inception(filter_size=filter[1], kernel=(3, 3), strides=stride[1])(x)
    
    y = Conv2D(64, (3, 3), strides=(2, 2), padding='same')(stem)
    y = Activation('relu')(y)

    z = MaxPooling2D(pool_size=(1, 1), strides=(2, 2))(stem)

    concat = concatenate([x, y, z])

    x = Flatten()(concat)

    for n in (128, 64):
        x = Dense(n, activation='relu', kernel_initializer='he_normal')(x)

    x = Dropout(0.5)(x)

    for n in (32, 16):
        x = Dense(n, activation='relu', kernel_initializer='he_normal')(x)

    x = Dropout(0.3)(x)
    
    output = Dense(classnum, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=output)

    LOSS = tf.keras.losses.categorical_crossentropy
    OPTIMIZER = tf.keras.optimizers.Nadam()

    model.compile(loss=LOSS, metrics=['acc'], optimizer=OPTIMIZER)

    return model
