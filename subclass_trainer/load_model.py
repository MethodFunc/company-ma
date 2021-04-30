from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation, Flatten, Conv2D, MaxPooling2D, Input
from tensorflow.keras.models import Model
import tensorflow as tf


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
    # create model
    input_layer = Input(shape=train_image[0].shape)
    x = BatchConvolution(16, (3, 3), 'relu', 'he_normal')(input_layer)
    x = BatchConvolution(32, (3, 3), 'relu', 'he_normal')(x)
    x = MaxPooling2D()(x)
    x = BatchConvolution(64, (3, 3), 'relu', 'he_normal')(x)
    x = BatchConvolution(128, (3, 3), 'relu', 'he_normal')(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu', kernel_initializer='he_normal')(x)
    x = Dense(64, activation='relu', kernel_initializer='he_normal')(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu', kernel_initializer='he_normal')(x)
    x = Dense(16, activation='relu', kernel_initializer='he_normal')(x)
    x = Dropout(0.3)(x)
    output = Dense(classnum, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=output)

    LOSS = tf.keras.losses.categorical_crossentropy
    OPTIMIZER = tf.keras.optimizers.Nadam()

    model.compile(loss=LOSS, metrics=['accuracy'], optimizer=OPTIMIZER)

    return model
