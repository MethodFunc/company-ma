from dataset_maker import load_dataset
from roi import setting_roi
import tensorflow.keras as keras
from datetime import datetime
import logging.config
import os
import tensorflow as tf
import sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.config.fileConfig('logging.conf')
logger = logging.getLogger('tf2trainer')


class Model_train:
    def __init__(self, SOURCE_PATH, CATEGORIES, SAMPLE_NUMBER, EPOCHS, BATCH_SIZE, ROI, LOAD_MODEL=None):
        self.SOURCE_PATH = SOURCE_PATH.replace('\\', '/')
        self.LOAD_MODEL = LOAD_MODEL
        self.CATEGORIES = CATEGORIES
        self.SAMPLE_NUMBER = SAMPLE_NUMBER
        self.EPOCHS = EPOCHS
        self.BATCH_SIZE = BATCH_SIZE
        self.ROI = setting_roi(ROI)
        self.history = None
        self.train_images = None
        self.train_labels = None
        self.test_images = None
        self.test_labels = None
        self.checkpoint = None
        self.height = None
        self.width = None
        self.depth = None

        self.CLASSNUM = len(CATEGORIES)

        self.classname = 'roadsurface'
        self.FRAME_WIDTH, self.FRAME_HEIGHT = 1520, 2688
        self.ROI_WIDTH, self.ROI_HEIGHT = 150, 150

        logger.info(f'>> Python version     : {sys.version}')
        logger.info(f'>> Tensorflow version : {tf.__version__}')
        logger.info(f'>> Keras version      : {keras.__version__}')

        self.start_time = datetime.now()

    def __call__(self):
        self.load_image_preprocessing()

        # Model Check-Point
        os.chdir(self.SOURCE_PATH)
        cp_path = f'cp_{self.start_time.strftime("%Y%m%d%H%M%S")}.tf'
        self.checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=cp_path, monitor='val_loss', verbose=3,
                                                             save_best_only=True)

        if not self.LOAD_MODEL:
            model = self.create_model()
            OPTIMIZER = tf.keras.optimizers.Adam(lr=0.001)
            model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
            self.model_fit(model)
            self.result_and_test_save()
            self.show_graph_plotly()

        if self.LOAD_MODEL:
            self.LOAD_MODEL = self.LOAD_MODEL.replace('\\', '/')
            model = tf.keras.models.load_model(self.LOAD_MODEL)
            # TEST 앞 4개의 층 동결 (추가학습 할때 가중치가 변하지 않음)
            base_model = tf.keras.models.Sequential(model.layers[:4])
            # base_model.summary()

            for layer in base_model.layers[:2]:
                layer.trainable = False
            # base_model.add(tf.keras.layers.GlobalAveragePooling2D(name='GAP'))
            # base_model.add(tf.keras.layers.Dense(self.CLASSNUM, activation='softmax', name='output_lapyer'))
            base_model.add(tf.keras.layers.Dropout(0.7, name='dropout_a'))
            base_model.add(tf.keras.layers.Flatten())
            base_model.add(tf.keras.layers.Dense(64, activation='relu', kernel_initializer='he_uniform', name='dense_a'))
            base_model.add(tf.keras.layers.Dropout(0.3, name='dropout_b'))
            base_model.add(tf.keras.layers.Dense(32, activation='relu', kernel_initializer='he_uniform', name='dense_b'))
            base_model.add(tf.keras.layers.Dropout(0.1, name='dropout_c'))
            base_model.add(tf.keras.layers.Dense(self.CLASSNUM, activation='softmax', name='dense_c'))
            base_model.summary()


            OPTIMIZER = tf.keras.optimizers.Adam(lr=0.0009)
            base_model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
            self.model_fit(base_model)
            self.result_and_test_save()
            self.show_graph_plotly()

            # 2차 재학습 모델 저장
            # recent_time = datetime.now()
            # cp_path = f'cp_{recent_time.strftime("%Y%m%d%H%M%S")}.tf'
            # self.checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=cp_path, monitor='val_loss', verbose=3,
            #                                                      save_best_only=True)
            #
            # # 한번 학습 후 동결 해제 하고 미세 조정 재학습.
            # for layer in base_model.layers[:3]:
            #     layer.trainable = True
            # OPTIMIZER = tf.keras.optimizers.Adam(lr=0.0005)
            # base_model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
            # self.model_fit(base_model)
            # self.result_and_test_save(base_model)
            # self.show_graph_plotly()

    # Data Loading
    def load_image_preprocessing(self):
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = \
            load_dataset(source_path=self.SOURCE_PATH, roi=self.ROI, sample_number=self.SAMPLE_NUMBER,
                         categories=self.CATEGORIES)
        (self.height, self.width, self.depth) = self.train_images[0].shape

        self.train_images = self.train_images / 255.
        self.test_images = self.test_images / 255.

        self.train_labels = tf.keras.utils.to_categorical(self.train_labels, self.CLASSNUM)
        self.test_labels = tf.keras.utils.to_categorical(self.test_labels, self.CLASSNUM)

    # 모델 수정시
    def create_model(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                                         kernel_initializer='he_uniform',
                                         input_shape=(self.height, self.width, self.depth)))

        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Dropout(0.7))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(64, activation='relu', kernel_initializer='he_uniform'))
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Dense(32, activation='relu', kernel_initializer='he_uniform'))
        model.add(tf.keras.layers.Dropout(0.1))
        model.add(tf.keras.layers.Dense(self.CLASSNUM, activation='softmax'))

        return model

    # Model fit
    def model_fit(self, model):
        try:
            self.history = model.fit(self.train_images, self.train_labels, epochs=self.EPOCHS,
                                     batch_size=self.BATCH_SIZE,
                                     validation_data=(self.test_images, self.test_labels), callbacks=[self.checkpoint],
                                     verbose=2)
        except Exception as ex:
            logger.error(f'{ex}.')
            exit()

    # result print and load_model test save
    def result_and_test_save(self, load_model=None):
        finish_time = datetime.now()
        run_time = finish_time - self.start_time

        loss_data = list(map(str, self.history.history['val_loss']))
        # print(f'loss_data >> {loss_data}')
        fname = list(map(str, self.history.history['val_loss']))[self.EPOCHS - 1]

        floss = np.min(self.history.history['val_loss'])
        facc = self.history.history['val_accuracy'][self.history.history['val_loss'].index(floss)] * 100
        fepoch = self.history.history['val_loss'].index(floss) + 1
        # print(f'fname >> {fname}')

        logger.info(f'>> Runtime     : {run_time}')
        logger.info(f'>> optimal val_loss >> {floss:.5f}({fepoch} epoch)')
        logger.info(f'>> optimal val_acc >> {facc:.2f}% ({fepoch} epoch)')

        logger.info('____________________________________________________________')
        if load_model:
            load_model.save(f'model_{self.start_time.strftime("%Y%m%d%H%M%S")}_{fname[2:6]}.tf')
            test_loss, test_acc = load_model.evaluate(self.test_images, self.test_labels, verbose=3)
            print(f'test_loss: {test_loss}')
            print(f'test_acc: {test_acc}')

    # Show Graphs - plotly
    def show_graph_plotly(self):
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss", "Accuracy"))

        fig.add_trace(go.Scatter(y=self.smooth_curve(loss), mode='lines', name='train loss'), row=1, col=1)
        fig.add_trace(go.Scatter(y=self.smooth_curve(val_loss), mode='lines', name='val loss'), row=1, col=1)

        fig.add_trace(go.Scatter(y=self.smooth_curve(acc), mode='lines', name='train acc'), row=1, col=2)
        fig.add_trace(go.Scatter(y=self.smooth_curve(val_acc), mode='lines', name='val acc'), row=1, col=2)

        fig.update_xaxes(title='Epoch', row=1, col=1)
        fig.update_xaxes(title='Epoch', row=1, col=2)

        fig.update_yaxes(title='Loss', row=1, col=1)
        fig.update_yaxes(title='Accuracy', row=1, col=2)
        fig.update_layout(height=720, title_text="Loss & Acc")

        fig.show()

    def smooth_curve(self, points, factor=0.8):
        smoothed_points = []
        for point in points:
            if smoothed_points:
                previous = smoothed_points[-1]
                smoothed_points.append(previous * factor + point * (1-factor))
            else:
                smoothed_points.append(point)
        return smoothed_points

if __name__ == '__main__':
    LOAD_MODEL = None
    # ---------- Change it -------------------------------
    SOURCE_PATH = r'F:\MK-SD53R\2021-02-04\202'  # Change Train image directory
    CATEGORIES = ['normal_day', 'normal_night']  # Change
    # 추가 학습을 진행하지 않을꺼면 주석처리해주세요.
    # LOAD_MODEL = r'C:\Users\ThreadRipper\Desktop\33A_new_etc_train\cp_20201225133551.tf'  # Change load_model
    SAMPLE_NUMBER = 250  # Less in Image Folder
    EPOCHS = 100
    BATCH_SIZE = 128
    # 201 = Road, 202 = Weather
    # 33A_201, 33A_201_old, 33A_201_fusion, 33A_202, 33C_201, 33C_201_fusion, 53R_201, 53R_202
    ROI = '53R_202'
    # ----------------------------------------------------

    if not LOAD_MODEL:
        start_fit = Model_train(SOURCE_PATH, CATEGORIES, SAMPLE_NUMBER, EPOCHS, BATCH_SIZE, ROI)
    if LOAD_MODEL:
        start_fit = Model_train(SOURCE_PATH, CATEGORIES, SAMPLE_NUMBER, EPOCHS, BATCH_SIZE, ROI, LOAD_MODEL)

    start_fit()
