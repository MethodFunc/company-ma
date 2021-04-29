from dataset_maker import load_dataset
from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
import keras
import logging.config
import os
import tensorflow as tf
import sys


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logging.config.fileConfig('logging.conf')
logger = logging.getLogger('tf2trainer')

logger.info(f'>> Python version     : {sys.version}')
logger.info(f'>> Tensorflow version : {tf.__version__}')
logger.info(f'>> Keras version      : {keras.__version__}')

classname = 'roadsurface'

SOURCE_PATH = 'D:/MKWS01/datasets/53R(Sunrise_Sunset)'
# SOURCE_PATH = 'C:/Users/ThreadRipper/Desktop/dry&wet_train/MK-SD33C/night'
# SOURCE_PATH = 'D:/MKWS01/python/tf2trainer/33A(sample)/Day'
FRAME_WIDTH, FRAME_HEIGHT = 1520, 2688
ROI_WIDTH, ROI_HEIGHT = 150, 150
ROI = []

# MK-SD33A ORG ROI(S11)
# ROI = [(2, 4), (3, 4), (1, 5), (2, 5), (3, 5), (0, 6), (1, 6), (2, 6), (0, 7), (1, 7), (2, 7), (0, 8), (1, 8), (0, 9), (0, 10)]

#

# MK-SD33A NEW ROI(S43)
# ROI = [(0, 5), (0, 7), (0, 8), (0, 9), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (2, 5), (2, 6), (2, 10), (3, 3), (3, 5), (3, 9), (4, 7)]


# MK-SD33A fusionROI(S11+S43)
# ROI = [(0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (1, 5), (1, 6), (1, 7), (1, 8), (2, 4), (2, 5), (2, 6), (2, 7), (3, 4), (3, 5), (4, 4)]

# MK-SD33C NEW ROI(S43)
#ROI = [(8, 6), (9, 7), (6, 8), (7, 8), (9, 8), (6, 9), (7, 9), (6, 10),
#        (7, 10), (8, 10), (6, 11), (7, 11), (8, 11), (6, 12), (7, 12), (8, 12), (9, 12),
#        (6, 13), (7, 13), (8, 13), (9, 13), (6, 14), (7, 14), (8, 14), (9, 14),
#        (6, 15), (7, 15), (8, 15), (9, 15), (6, 16), (7, 16), (8, 16), (9, 16)]

# MK-SD33C fusionROI(S11+S43)
# ROI = [(6, 8), (6, 9), (6, 10), (6, 11), (6, 12), (6, 13), (6, 14), (6, 15), (7, 9), (7, 10), (7, 11), (7, 12), (7, 13), (7, 14), (7, 15), (8, 11), (8, 12), (8, 13), (8, 14), (8, 15), (9, 8), (9, 13), (9, 14), (9, 15)]

# MK-SD53R(weather_roi)(202)
# ROI = [(2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2), (7, 2), (8, 2), (9, 2)]

# MK-SD53R(road_roi)(201)
ROI = [(0, 9), (0, 10), (0, 11), (0, 12), (1, 8), (1, 9), (1, 10), (1, 14), (1, 15), (2, 7), (2, 8), (2, 12), (2, 13), (2, 14), (2, 15), (3, 6), (3, 10), (3, 11), (3, 12), (3, 13), (3, 14), (3, 15), (4, 4), (4, 8), (4, 9), (4, 10), (4, 11), (4, 12), (4, 13), (4, 14), (4, 15), (5, 3), (5, 7), (5, 8), (5, 9), (5, 10), (5, 11), (6, 5), (6, 6), (6, 7), (6, 8), (7, 3), (7, 4)]

CATEGORIES = ['dry', 'wet']
SAMPLE_NUMBER = 2192

# samples = 1000
(train_images, train_labels), (test_images, test_labels) =\
    load_dataset(source_path=SOURCE_PATH, roi=ROI, sample_number=SAMPLE_NUMBER, categories=CATEGORIES)  # , numberofsamples=39000)
(height, width, depth) = train_images[0].shape

# train_images = train_images.reshape(train_images.shape[0], height, width, depth)
# test_images = test_images.reshape(test_images.shape[0], height, width, depth)

train_images = train_images / 255.
test_images = test_images / 255.

epochs = 100
batch_size = 128
classnum = 2

train_labels = keras.utils.to_categorical(train_labels, classnum)
test_labels = keras.utils.to_categorical(test_labels, classnum)

def create_model():
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=(height, width, depth)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.7))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.1))
    # model.add(Dense(32, activation='relu'))
    model.add(Dense(classnum, activation='softmax'))
    # model.summary()
    # model.load_weights('C:/Users/ThreadRipper/Desktop/dry&wet_train/color_mode_33A_data/20200910_Dry,Wet_data/day/cp_20200910172142.h5')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

start_time = datetime.now()
# Model Check-Point
os.chdir(SOURCE_PATH)
cp_path = f'cp_{start_time.strftime("%Y%m%d%H%M%S")}.tf'
checkpoint = ModelCheckpoint(filepath=cp_path, monitor='val_loss', verbose=3, save_best_only=True)
load_model = tf.keras.models.load_model('C:/Users/ThreadRipper/Desktop/dry&wet_train/MK-SD53R_road&weather/road/cp_20201208093837.tf')
# Model Fit
try:
    #model = create_model()
    history = load_model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size,
                        validation_data=(test_images, test_labels), callbacks=[checkpoint], verbose=2)
except Exception as ex:
    logger.error(f'{ex}.')
    exit()

finish_time = datetime.now()
run_time = finish_time - start_time

# Model Saved
logger.info(f'>> Runtime     : {run_time}')

loss_data = list(map(str, history.history['val_loss']))
# print(f'loss_data >> {loss_data}')
fname = list(map(str, history.history['val_loss']))[epochs-1]
# print(f'fname >> {fname}')

logger.info(f'>> latest val_loss >> {fname[:6]}')
logger.info('____________________________________________________________')
load_model.save(f'model_{start_time.strftime("%Y%m%d%H%M%S")}_{fname[2:6]}.tf')

test_loss, test_acc = load_model.evaluate(test_images, test_labels, verbose=3)
print(f'test_loss: {test_loss}')

# ==============================================================================
'''
SOURCE_PATH = 'C:/Users/ThreadRipper/Desktop/dry&wet_train/color_mode_33A_data/20200910_Dry,Wet_data/night'
SAMPLE_NUMBER = 8010

# samples = 1000
(train_images, train_labels), (test_images, test_labels) =\
    load_dataset(source_path=SOURCE_PATH, roi=ROI1, sample_number=SAMPLE_NUMBER)  # , numberofsamples=39000)
(height, width, depth) = train_images[0].shape

# train_images = train_images.reshape(train_images.shape[0], height, width, depth)
# test_images = test_images.reshape(test_images.shape[0], height, width, depth)

train_images = train_images / 255.
test_images = test_images / 255.

epochs = 100
batch_size = 128
classnum = 2

train_labels = keras.utils.to_categorical(train_labels, classnum)
test_labels = keras.utils.to_categorical(test_labels, classnum)


model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=(height, width, depth)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.7))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.1))
# model.add(Dense(32, activation='relu'))
model.add(Dense(classnum, activation='softmax'))
# model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

start_time = datetime.now()
# Model Check-Point
os.chdir(SOURCE_PATH)
cp_path = f'cp_{start_time.strftime("%Y%m%d%H%M%S")}.h5'
checkpoint = ModelCheckpoint(filepath=cp_path, monitor='val_loss', verbose=3, save_best_only=True)

# Model Fit
try:
    history = model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size,
                        validation_data=(test_images, test_labels), callbacks=[checkpoint], verbose=2)
except Exception as ex:
    logger.error(f'{ex}.')
    exit()

finish_time = datetime.now()
run_time = finish_time - start_time

# Model Saved
logger.info(f'>> Runtime     : {run_time}')

loss_data = list(map(str, history.history['val_loss']))
# print(f'loss_data >> {loss_data}')
fname = list(map(str, history.history['val_loss']))[epochs-1]
# print(f'fname >> {fname}')

logger.info(f'>> latest val_loss >> {fname[:6]}')
logger.info('____________________________________________________________')
model.save(f'model_{start_time.strftime("%Y%m%d%H%M%S")}_{fname[2:6]}.h5')

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=3)
print(f'test_loss: {test_loss}')
'''