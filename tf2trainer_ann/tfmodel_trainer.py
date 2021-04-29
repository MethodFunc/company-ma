from dataset_maker import load_dataset
from roi import setting_roi
from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
import keras
import logging.config
import numpy as np
import os
import tensorflow as tf
import sys
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# %%
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

classname = 'roadsurface'

FRAME_WIDTH, FRAME_HEIGHT = 1520, 2688
ROI_WIDTH, ROI_HEIGHT = 150, 150
ROI = []
# --------------------Change It-----------------------
SOURCE_PATH = r'F:\MK-SD53R\2021-01-08\201'  # Change directory
CATEGORIES = ['dry_day', 'dry_night']

SAMPLE_NUMBER = 10  # Less in Image Folder
EPOCHS = 10
BATCH_SIZE = 128

# 201 = Road, 202 = Weather
# 33A_201, 33A_201_old, 33A_201_fusion, 33C_201, 33C_201_fusion, 53R_201, 53R_202
ROI = setting_roi('53R_201')
# ----------------------------------------------------
# %%
SOURCE_PATH = SOURCE_PATH.replace('\\', '/')
CLASSNUM = len(CATEGORIES)
# %%
print(CLASSNUM)
# %%
# Load_data
(train_images, train_labels), (test_images, test_labels) = \
    load_dataset(source_path=SOURCE_PATH, roi=ROI, sample_number=SAMPLE_NUMBER, categories=CATEGORIES)

(height, width, depth) = train_images[0].shape

# Preprocessing data
train_images = train_images / 255.
test_images = test_images / 255.

train_labels = keras.utils.to_categorical(train_labels, CLASSNUM)
test_labels = keras.utils.to_categorical(test_labels, CLASSNUM)


train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(32)

tf.compat.v1
tf.nn.dropout()
class MyCnnModel(tf.keras.Model):
    def __init__(self):
        super(MyCnnModel, self).__init__()
        self.conv1 = Conv2D(16, kernel_size=(3, 3), activation='relu')
        self.max = MaxPooling2D()
        self.conv2 = Conv2D(32, kernel_size=(3, 3), activation='relu')
        self.do1 = Dropout(0.7)
        self.flatten = Flatten()
        self.d1 = Dense(64, activation='relu')
        self.do2 = Dropout(0.3)
        self.d2 = Dense(32, activation='relu')
        self.do3 = Dropout(0.1)
        self.d3 = Dense(CLASSNUM, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.max(x)
        x = self.conv2(x)
        x = self.max(x)
        x = self.flatten(x)
        x = self.do1(x)
        x = self.d1(x)
        x = self.do2(x)
        x = self.d2(x)
        x = self.do3(x)

        return self.d3(x)


model = MyCnnModel()

# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
#
#
# history = model.fit(train_images, train_labels, epochs=10)

loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_acc')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_acc')


@tf.function
def train_step(images1, labels1):
    with tf.GradientTape() as tape:
        predictions = model(images1, training=True)
        loss = loss_object(labels1, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels1, predictions)


@tf.function
def test_step(images2, labels2):
    predictions = model(images2, training=False)
    t_loss = loss_object(labels2, predictions)
    test_loss(t_loss)
    test_accuracy(labels2, predictions)


test_lossl = []

for epoch in range(EPOCHS):
    for images, labels in train_ds:
        train_step(images, labels)

    for test_img, test_label in test_ds:
        test_step(test_img, test_label)

    # tf.experimental.numpy.append(test_lossl, test_loss.result(), axis=0)
    # tf.concat([test_lossl, test_loss.result()], axis=0)

    if len(test_lossl) == 0:
        print(f'First save : loss - {test_loss.result()} & acc - {test_accuracy.result()*100}')
        model.save("my_model")

    # print(f'{test_loss.result().numpy() < min(test_lossl)}, test_loss:{test_loss.result().numpy()}, min_test : {min(test_lossl)}')

    elif test_loss.result() < min(test_lossl):
        print(f'Save! : loss - {test_loss.result()} & acc - {test_accuracy.result()*100}')
        model.save("my_model")

    print(f'Epoch {epoch + 1}, '
          f'train_loss :{train_loss.result()}, '
          f'train_acc : {train_accuracy.result() * 100}, '
          f'test_loss = {test_loss.result()}, '
          f'test_acc : {test_accuracy.result() * 100}')
#
    test_lossl.append(test_loss.result())
# # Create Model --- Change it -------------------------------------------------------------------------------------------
# def create_model():
#     model = Sequential()
#     model.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=(height, width, depth)))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.7))
#     model.add(Flatten())
#     model.add(Dense(64, activation='relu'))
#     model.add(Dropout(0.3))
#     model.add(Dense(32, activation='relu'))
#     model.add(Dropout(0.1))
#     model.add(Dense(CLASSNUM, activation='softmax'))
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
#     return model
# # -----------------------------------------------------------------------------------------------------------------------
# start_time = datetime.now()
#
# # Model Check-Point
# os.chdir(SOURCE_PATH)
# cp_path = f'cp_{start_time.strftime("%Y%m%d%H%M%S")}.tf'
# checkpoint = ModelCheckpoint(filepath=cp_path, monitor='val_loss', verbose=3, save_best_only=True)
# model, load_model = False, False
#
# # Model select ------------------- Change it------------------------------------
# model = create_model()
# # load_model = tf.keras.models.load_model('C:/Users/ThreadRipper/Desktop/dry&wet_train/MK-SD53R_road&weather/road/cp_20201208093837.tf')
# # -------------------------------------------------------------------------------
#
# # Model Fitooo
# def model_fit(x):
#     try:
#         history = x.fit(train_images, train_labels, epochs=EPOCHS, batch_size=BATCH_SIZE,
#                         validation_data=(test_images, test_labels), callbacks=[checkpoint], verbose=2)
#     except Exception as ex:
#         logger.error(f'{ex}.')
#         exit()
#
#     return history
#
# def result_and_test_save(x):
#     finish_time = datetime.now()
#     run_time = finish_time - start_time
#
#     # Model Saved
#     logger.info(f'>> Runtime     : {run_time}')
#
#     loss_data = list(map(str, history.history['val_loss']))
#     # print(f'loss_data >> {loss_data}')
#     fname = list(map(str, history.history['val_loss']))[EPOCHS - 1]
#     facc = list(map(str, history.history['val_accuracy']))[EPOCHS - 1]
#
#     # print(f'fname >> {fname}')
#
#     logger.info(f'>> latest val_loss >> {fname[:6]}')
#     logger.info(f'>> latest val_acc >> {facc[:6]}')
#     logger.info('____________________________________________________________')
#
#     x.save(f'model_{start_time.strftime("%Y%m%d%H%M%S")}_{fname[2:6]}.tf')
#     test_loss, test_acc = x.evaluate(test_images, test_labels, verbose=3)
#     print(f'test_loss: {test_loss}')
#     print(f'test_acc: {test_acc}')
#
#
# if load_model:
#     history = model_fit(load_model)
#     result_and_test_save(load_model)
#
# # Show Graphs - matplotlib
# def show_Graph_matplotlib():
#     plt.Figure(figsize=(11, 12))
#
#     plt.subplot(211)
#     plt.plot(history.history['loss'], 'y', label='train loss')
#     plt.plot(history.history['val_loss'], 'r', label='val loss')
#     plt.grid(True)
#     plt.xlabel('EPOCH')
#     plt.ylabel('loss')
#     plt.legend(loc='best')
#
#     plt.subplot(212)
#     plt.plot(history.history['accuracy'], 'b', label='train acc')
#     plt.plot(history.history['val_accuracy'], 'g', label='val acc')
#     plt.grid(True)
#     plt.xlabel('EPOCH')
#     plt.ylabel('acc')
#     plt.legend(loc='best')
#
#     plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.5)
#
#     plt.show()
#
# # Show Graphs - plotly
# def show_graph_plotly():
#     fig = make_subplots(rows=1 , cols=2, subplot_titles=("Loss", "Accuracy"))
#
#     fig.add_trace(go.Scatter(y = history.history['loss'], mode='lines', name='train loss'), row=1, col=1)
#     fig.add_trace(go.Scatter(y = history.history['val_loss'], mode='lines', name= 'val loss'), row=1, col=1)
#
#     fig.add_trace(go.Scatter(y = history.history['accuracy'], mode='lines', name='train acc'), row=1, col=2)
#     fig.add_trace(go.Scatter(y = history.history['val_accuracy'], mode='lines', name='val acc'), row=1, col=2)
#
#     fig.update_xaxes(title='Epoch', row=1, col=1)
#     fig.update_xaxes(title='Epoch', row=1, col=2)
#
#     fig.update_yaxes(title='Loss', row=1, col=1)
#     fig.update_yaxes(title='Accuracy', row=1, col=2)
#     fig.update_layout(height=720, title_text="Loss & Acc")
#
#     fig.show()


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
