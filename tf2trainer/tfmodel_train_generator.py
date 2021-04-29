import tensorflow as tf
import tensorflow.keras.backend as K
import os, fnmatch

from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate, BatchNormalization, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, \
    Dense, Input
from datetime import datetime
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

class Mish(tf.keras.layers.Activation):
    def __init__(self, activation, **kwargs):
        super(Mish, self).__init__(activation, **kwargs)
        self.__name__ = 'Mish'


def mish(x):
    return x * K.tanh(K.softplus(x))


HEIGHT, WIDTH, DEPTH = 128, 128, 3
BATCH_SIZE = 64
SOURCE_PATH = r'F:\MK-SD53R\2021-02-03\202'
TEST_SOURCE_PATH = r'F:\MK-SD53R\2021-02-04\202'

SOURCE_PATH = SOURCE_PATH.replace('\\', '/')
TEST_SOURCE_PATH = TEST_SOURCE_PATH.replace('\\', '/')

CATEGORIES = ['normal_day', 'normal_night']


CLASSNUM = len(CATEGORIES)

start_time = datetime.now()


log = "./logs/"
cp_path = f'cp_{start_time.strftime("%Y%m%d%H%M%S")}.tf'

checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=cp_path, monitor='val_loss', verbose=3,
                                                save_best_only=True)

if not os.path.isdir(log):
    os.mkdir(log)

log_dir = log + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


get_custom_objects().update({'mish': Mish(mish)})

def datagen_generator():
    datagen = ImageDataGenerator(width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, fill_mode='nearest')
    for categories in CATEGORIES:
        file_list = fnmatch.filter(os.listdir(SOURCE_PATH), f'{categories}/*.jpg')

        for img_file in file_list:
            img = load_img(img_file)
            x = img_to_array(img)

            i = 0
            for batch in datagen.flow(x, batch_size=1, save_to_dir=f'{SOURCE_PATH}/preview', save_prefix=categories, save_format='jpg'):
                i += 1

                if i > 50:
                    break





train_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)
test_datagen = ImageDataGenerator(rescale=1. / 255)



train_set = train_datagen.flow_from_directory(SOURCE_PATH,
                                              target_size=(HEIGHT, WIDTH), batch_size=BATCH_SIZE, classes=CATEGORIES, class_mode='categorical',
                                              subset='training')
validation_set = train_datagen.flow_from_directory(SOURCE_PATH,
                                                   target_size=(HEIGHT, WIDTH), batch_size=BATCH_SIZE, classes=CATEGORIES, class_mode='categorical',
                                                   subset='validation')
test_set = test_datagen.flow_from_directory(TEST_SOURCE_PATH,
                                            target_size=(HEIGHT, WIDTH), classes=CATEGORIES, class_mode='categorical')

# file_writer = tf.summary.create_file_writer(log_dir)
# with file_writer.as_default():
#     tf.summary.image("Sample Train Image", train_set, max_outputs=10, step=0)

input_layer = Input(shape=(HEIGHT, WIDTH, DEPTH))

x = Conv2D(32, kernel_size=(3, 3), padding='same', name='conv_1')(input_layer)
x = BatchNormalization()(x)
x = Activation('mish')(x)

x = Conv2D(64, kernel_size=(3, 3), padding='same', name='conv_2')(x)
x = BatchNormalization()(x)
x = Activation('mish')(x)

mix = concatenate([input_layer, x])

mp = MaxPooling2D(2)(mix)

x = Conv2D(32, kernel_size=(3, 3), name='conv_3')(mp)
x = BatchNormalization()(x)
x = Activation('mish')(x)

x = Conv2D(64, kernel_size=(3, 3), name='conv_4')(x)
x = BatchNormalization()(x)
x = Activation('mish')(x)

y = Conv2D(32, kernel_size=(3, 3), name='conv_5')(mp)
y = BatchNormalization()(y)
y = Activation('mish')(y)

y = Conv2D(64, kernel_size=(3, 3), name='conv_6')(y)
y = BatchNormalization()(y)
y = Activation('mish')(y)

x = MaxPooling2D(2)(x)
y = MaxPooling2D(2)(y)
mix2 = concatenate([x, y])

x = MaxPooling2D(2)(mix2)

x = Flatten()(x)
x = Dropout(0.5)(x)
x = Dense(512)(x)
x = Activation('mish')(x)
x = Dropout(0.5)(x)
x = Dense(256)(x)
x = Activation('mish')(x)
x = Dropout(0.3)(x)
x = Dense(64)(x)
x = Activation('mish')(x)
x = Dropout(0.3)(x)
x = Dense(CLASSNUM)(x)
outputs = Activation('softmax')(x)

model = Model(inputs=input_layer, outputs=outputs)

OPTIMIZER = tf.keras.optimizers.Adam(lr=0.002)

model.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(train_set, steps_per_epoch=train_set.samples // BATCH_SIZE, epochs=10, validation_data=validation_set,
                    validation_steps=validation_set.samples // BATCH_SIZE, callbacks=[tensorboard_callback], verbose=1)
