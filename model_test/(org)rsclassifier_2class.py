import cv2, glob
import os, time
import time
import random
import threading
import numpy as np
from keras.models import load_model
from datetime import datetime, timedelta
from collections import deque
from roi import setting_roi
# import video_control as vc
import fnmatch
import tensorflow as tf
import math
import shutil
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from termcolor import colored

# import logging.config

# logging.config.fileConfig('logging.conf')
# logger = logging.getLogger('rsclassifier')

'''
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
tf.keras.backend.set_session(tf.Session(config=config))
'''


class Classifier:
    def __init__(self, MODEL_PATH1, MODEL_PATH2, PATH, CATEGORIES, CATEGORIES2, SAMPLING_RATIO, SHOW=None, MOVE=None):
        self.MODEL_PATH1 = MODEL_PATH1
        self.MODEL_PATH2 = MODEL_PATH2
        self.PATH = PATH
        self.CATEGORIES = CATEGORIES
        self.CATEGORIES2 = CATEGORIES2
        self.SAMPLING_RATIO = SAMPLING_RATIO
        self.ROI1 = [(5, 0), (6, 0), (7, 0), (1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (3, 2),  (4, 2), (5, 2), (6, 2),  (9, 3), (9, 4), (9, 5)]
        self.ROI2 = [(1, 10), (2, 8), (2, 9), (2, 10), (3, 7), (3, 8), (3, 9), (4, 7), (4, 10), (4, 11), (5, 9), (5, 10), (5, 11), (6, 8), (6, 9), (6, 10)]
        self.SHOW = SHOW
        self.MOVE = MOVE

        self.MODEL_PATH1 = self.MODEL_PATH1.replace('\\', '/')
        self.MODEL_PATH2 = self.MODEL_PATH2.replace('\\', '/')
        self.PATH = self.PATH.replace('\\', '/')
        self.ROI_WIDTH, self.ROI_HEIGHT = 150, 150

        self.model_1 = tf.keras.models.load_model(self.MODEL_PATH1)  # Model
        self.model_2 = tf.keras.models.load_model(self.MODEL_PATH2)  # Model
        self.PREDICT_COUNT = len(CATEGORIES)
        self.PREDICTION_INTERVAL = 1
        self.PREDICTION_QUEUE_LENGTH = 20
        self.STATUS_SENSITIVITY = 20
        self.TODAY = datetime.today()

    def __call__(self, *args, **kwargs):
        self.predict_roadsurface()

    def brightness_calc(self, roi_img):
        lab = cv2.cvtColor(roi_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        y, x, z = roi_img.shape
        l_blur = cv2.GaussianBlur(l, (11, 11), 5)
        maxval = []

        count_percent = 1
        count_percent = count_percent / 100
        row_percent = int(count_percent * x)
        column_percent = int(count_percent * y)

        for i in range(1, x - 1):
            if i % row_percent == 0:
                for j in range(1, y - 1):
                    if j % column_percent == 0:
                        img_segment = l_blur[i:i + 3, j:j + 3]
                        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(img_segment)
                        maxval.append(maxVal)

        avg_maxval = round(sum(maxval) / len(maxval))

        return avg_maxval

    def predict_roadsurface(self, **kwargs):

        images = glob.glob(f'{self.PATH}/*.jpg')
        count, dd_count, dn_count, wd_count, wn_count, etc_count, etc2_count = 0, 0, 0, 0, 0, 0, 0
        accruacy = []
        for imgs in images:
            filedir = os.path.basename(imgs)
            filename = os.path.splitext(filedir)
            img = cv2.imread(imgs)
            roi_set1, roi_set2, roi_set2_rgb = [], [], []
            for (i, j) in self.ROI1:
                x, y = i * self.ROI_WIDTH, j * self.ROI_HEIGHT
                roi_img = img[y:y + self.ROI_HEIGHT, x:x + self.ROI_WIDTH]
                roi_set1.append(roi_img)

            for (i, j) in self.ROI2:
                x, y = i * self.ROI_WIDTH, j * self.ROI_HEIGHT
                roi_img = img[y:y + self.ROI_HEIGHT, x:x + self.ROI_WIDTH]

                # b = roi_img[:, :, 0].mean()
                # g = roi_img[:, :, 1].mean()
                # r = roi_img[:, :, 2].mean()
                # bright = self.brightness_calc(roi_img)
                # mean = roi_img.mean()
                # merged = np.hstack([mean, bright])


                # roi_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
                # ret, roi_img = cv2.threshold(roi_img, 142, 255, cv2.THRESH_BINARY)
                # k = 15
                # C = 20
                #
                # th2 = cv2.adaptiveThreshold(roi_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, k, C)
                # roi_set2_rgb.append(bright)
                roi_set2.append(roi_img)



            sampling_number = min(len(roi_set1), self.SAMPLING_RATIO)
            roi_set1 = random.sample(roi_set1, sampling_number)
            roi_set1 = np.array(roi_set1).reshape(-1, self.ROI_HEIGHT, self.ROI_WIDTH, 3)
            roi_set1 = roi_set1 / 255.0

            sampling_number = min(len(roi_set2), self.SAMPLING_RATIO)
            roi_set2 = random.sample(roi_set2, sampling_number)
            roi_set2 = np.array(roi_set2).reshape(-1, self.ROI_HEIGHT, self.ROI_WIDTH, 3)
            roi_set2 = roi_set2 / 255.0
            # merged = np.array(roi_set2_rgb).astype(np.float32)
            # merged = merged / 255.0
            '''
            for a in range(len(roi_set)):
                cv2.imshow('asd', roi_set[a])
                cv2.waitKey(0) == 0xFF
                print(np.int64(model_day.predict(roi_set)*10))
            '''
            # test = np.hstack([roi_set2, merged])
            pred_values_1 = self.model_1.predict(roi_set1)  # if is_daytime else list(model_night.predict(roi_set))
            pred_values_2 = self.model_2.predict(roi_set2)  # if is_daytime else list(model_night.predict(roi_set))

            # pred_values_2 = self.model_2.predict([roi_set2, merged])  # if is_daytime else list(model_night.predict(roi_set))
            # print(np.int64(pred_values*10))
            preds_1, preds_2 = [], []
            for pred in pred_values_1:
                preds_1.append(np.argmax(pred))

            for pred in pred_values_2:
                preds_2.append(np.argmax(pred))

            prediction_list_1, prediction_list_2  = [], []
            for i in range(self.PREDICT_COUNT):
                prediction_list_1.append(preds_1.count(i))
                prediction_list_2.append(preds_2.count(i))

            # print('filename :', filename[0], 'pred :', preds, 'count :', prediction_list, 'class :', np.argmax(prediction_list), 'category :', CATEGORIES[np.argmax(prediction_list)])

            # print('filename :', filename[0], 'pred :', preds, 'count :', prediction_list, 'class :', np.argmax(prediction_list), 'category :', CATEGORIES[np.argmax(prediction_list)])
            #
            # print(f'filename : {filename[0]}, count_a : {prediction_list_1}, count_b : {prediction_list_2}, class_a : {self.CATEGORIES[np.argmax(prediction_list_1)]}, class_b : {self.CATEGORIES2[np.argmax(prediction_list_2)]}')

            # if np.argmax(prediction_list_2) == 0:
            #     count += 1
            #     print(f'{count}, filename : {filename[0]}, {self.CATEGORIES2[np.argmax(prediction_list_2)]}, {colored(f"Result = {self.CATEGORIES2[np.argmax(prediction_list_2)]}", color ="red", on_color="on_grey")}')
            # elif np.argmax(prediction_list_2) == 1:
            #     dd_count += 1
            #     print(f'{dd_count} filename : {filename[0]}, {self.CATEGORIES2[np.argmax(prediction_list_2)]}, {colored(f"Result = {self.CATEGORIES2[np.argmax(prediction_list_2)]}", color="blue", on_color="on_grey")}')

            if np.argmax(prediction_list_1) == 0 and np.argmax(prediction_list_2) == 0:
                if not os.path.isdir(f'{self.PATH}/normal'):
                    os.mkdir(f'{self.PATH}/normal')
                shutil.move(f'{self.PATH}/{filename[0]}.jpg', f'{self.PATH}/normal/{filename[0]}.jpg')
                print(f'filename : {filename[0]}, class_a : {self.CATEGORIES[np.argmax(prediction_list_1)]}, class_b : {self.CATEGORIES2[np.argmax(prediction_list_2)]}, {colored("Result = Normal", color ="red", on_color="on_grey")}')
            elif np.argmax(prediction_list_1) == 0 and np.argmax(prediction_list_2) == 1:
                if not os.path.isdir(f'{self.PATH}/snow'):
                    os.mkdir(f'{self.PATH}/snow')
                shutil.move(f'{self.PATH}/{filename[0]}.jpg', f'{self.PATH}/snow/{filename[0]}.jpg')
                print(f'filename : {filename[0]}, class_a : {self.CATEGORIES[np.argmax(prediction_list_1)]}, class_b : {self.CATEGORIES2[np.argmax(prediction_list_2)]}, {colored("Result = Snow", color ="red", on_color="on_grey")}')
            elif np.argmax(prediction_list_1) == 1 and np.argmax(prediction_list_2) == 0:
                if not os.path.isdir(f'{self.PATH}/fog'):
                    os.mkdir(f'{self.PATH}/fog')
                shutil.move(f'{self.PATH}/{filename[0]}.jpg', f'{self.PATH}/fog/{filename[0]}.jpg')
                print(f'filename : {filename[0]}, class_a : {self.CATEGORIES[np.argmax(prediction_list_1)]}, class_b : {self.CATEGORIES2[np.argmax(prediction_list_2)]}, {colored("Result = Fog", color ="red", on_color="on_grey")}')
            elif np.argmax(prediction_list_1) == 1 and np.argmax(prediction_list_2) == 1:
                if not os.path.isdir(f'{self.PATH}/snow'):
                    os.mkdir(f'{self.PATH}/snow')
                shutil.move(f'{self.PATH}/{filename[0]}.jpg', f'{self.PATH}/snow/{filename[0]}.jpg')
                print(f'filename : {filename[0]}, class_a : {self.CATEGORIES[np.argmax(prediction_list_1)]}, class_b : {self.CATEGORIES2[np.argmax(prediction_list_2)]}, {colored("Result = Snow", color ="red", on_color="on_grey")}')

                # print(f'filename : {filename[0]}, count : {np.int64(pred_values*10)}, class : {CATEGORIES[np.argmax(prediction_list)]}')
                # print(len(images))
                # acc = (np.max(prediction_list) / np.sum(prediction_list)) * 100
                # accruacy.append(acc)
                # count += 1
                # print(f'전체 이미지 장수 : {len(images)}, 진행 이미지 장수 : {count}, 검증율 : {acc:.2f}%')


                # if self.SHOW is None:
                #     if np.argmax(prediction_list) == 0:
                #         dd_count += 1
                #         if self.MOVE:
                #             if not os.path.isdir(f'{self.PATH}/{self.CATEGORIES[0]}'):
                #                 os.mkdir(f'{self.PATH}/{self.CATEGORIES[0]}')
                #             shutil.move(f'{self.PATH}/{filename[0]}.jpg', f'{self.PATH}/{self.CATEGORIES[0]}/{filename[0]}.jpg')
                #
                #     elif np.argmax(prediction_list) == 1:
                #         dn_count += 1
                #         if self.MOVE:
                #             if not os.path.isdir(f'{self.PATH}/{self.CATEGORIES[1]}'):
                #                 os.mkdir(f'{self.PATH}/{self.CATEGORIES[1]}')
                #             shutil.move(f'{self.PATH}/{filename[0]}.jpg', f'{self.PATH}/{self.CATEGORIES[1]}/{filename[0]}.jpg')
                #     elif np.argmax(prediction_list) == 2:
                #         wd_count += 1
                #         if self.MOVE:
                #             if not os.path.isdir(f'{self.PATH}/{self.CATEGORIES[2]}'):
                #                 os.mkdir(f'{self.PATH}/{self.CATEGORIES[2]}')
                #             shutil.move(f'{self.PATH}/{filename[0]}.jpg', f'{self.PATH}/{self.CATEGORIES[2]}/{filename[0]}.jpg')
                #     elif np.argmax(prediction_list) == 3:
                #         wn_count += 1
                #         if self.MOVE:
                #             if not os.path.isdir(f'{self.PATH}/{self.CATEGORIES[3]}'):
                #                 os.mkdir(f'{self.PATH}/{self.CATEGORIES[3]}')
                #             shutil.move(f'{self.PATH}/{filename[0]}.jpg', f'{self.PATH}/{self.CATEGORIES[3]}/{filename[0]}.jpg')
                #     elif np.argmax(prediction_list) == 4:
                #         etc_count += 1
                #         if self.MOVE:
                #             if not os.path.isdir(f'{self.PATH}/{self.CATEGORIES[4]}'):
                #                 os.mkdir(f'{self.PATH}/{self.CATEGORIES[4]}')
                #             shutil.move(f'{self.PATH}/{filename[0]}.jpg', f'{self.PATH}/{self.CATEGORIES[4]}/{filename[0]}.jpg')
                #
                #     elif np.argmax(prediction_list) == 5:
                #         etc2_count += 1
                #         if self.MOVE:
                #             if not os.path.isdir(f'{self.PATH}/{self.CATEGORIES[5]}'):
                #                 os.mkdir(f'{self.PATH}/{self.CATEGORIES[5]}')
                #             shutil.move(f'{self.PATH}/{filename[0]}.jpg', f'{self.PATH}/{self.CATEGORIES[5]}/{filename[0]}.jpg')

        #         full_image = len(images)
        #
        #         dry_day = dd_count / full_image * 100
        #         dry_night = dn_count / full_image * 100
        #         wet_day = wd_count / full_image * 100
        #         wet_night = wn_count / full_image * 100
        #         etc = etc_count / full_image * 100
        #         etc2 = etc2_count / full_image * 100
        #
        #         ing_count = dd_count + dn_count + wd_count + wn_count + etc_count
        #         ing_percent = ing_count / full_image * 100
        #
        #         print(f'총 이미지 : {full_image}, 진행 갯수 : {ing_count}, 진행율 : {ing_percent:.2f}% - {CATEGORIES[0]} : {dry_day:.2f}%, {CATEGORIES[1]} : {dry_night:.2f}%' \
        #               f', {CATEGORIES[2]} : {wet_day:.2f}%,  {CATEGORIES[3]} : {wet_night:.2f}%, {CATEGORIES[4]} : {etc:.2f}%, {CATEGORIES[5]} : {etc2:.2f}%, filename : {filename[0]}', flush=True)
        #
        # print(f'총정확도: {(len(images) - count) / len(images) * 100: .2f} %')
        # if self.SHOW is not None:
        #     predict_percent = np.sum(accruacy) / len(accruacy)
        #     print(f'{predict_percent:.2f}')
            # print(f'전체 이미지 장수 : {len(images)}, 분류 이미지 장수 : {count}, 검증율 : {acc:.2f}%')

            # if np.argmax(prediction_list) == 0:
            #     if not os.path.isdir(f'{path}/dry_day'):
            #         os.mkdir(f'{path}/dry_day')
            #     dd_count += 1
            #
            #     shutil.move(f'{path}/{filename[0]}.jpg', f'{path}/dry_day/{filename[0]}.jpg')
            # elif np.argmax(prediction_list) == 1:
            #     if not os.path.isdir(f'{path}/dry_night'):
            #         os.mkdir(f'{path}/dry_night')
            #     dn_count += 1
            #     shutil.move(f'{path}/{filename[0]}.jpg', f'{path}/dry_night/{filename[0]}.jpg')
            # elif np.argmax(prediction_list) == 2:
            #     if not os.path.isdir(f'{path}/wet_day'):
            #         os.mkdir(f'{path}/wet_day')
            #     wd_count += 1
            #     shutil.move(f'{path}/{filename[0]}.jpg', f'{path}/wet_day/{filename[0]}.jpg')
            # elif np.argmax(prediction_list) == 3:
            #     if not os.path.isdir(f'{path}/wet_night'):
            #         os.mkdir(f'{path}/wet_night')
            #     wn_count += 1
            #     shutil.move(f'{path}/{filename[0]}.jpg', f'{path}/wet_night/{filename[0]}.jpg')

            # print(f'filename : {filename[0]}, count : {prediction_list}, class : {CATEGORIES[np.argmax(prediction_list)]}')
            # # print(f'filename : {filename[0]}, count : {np.int64(pred_values*10)}, class : {CATEGORIES[np.argmax(prediction_list)]}')
            # print(len(images))
            # acc = (np.max(prediction_list) / np.sum(prediction_list)) * 100
            # print(f'{acc}%')

            # fig = make_subplots(rows = 1 , cols=2, subplot_titles=['Graph', 'Histogram'])
            #
            # fig.add_trace(go.Scatter(y=accruacy, mode='lines', name='Acc'), row=1, col=1)
            # fig.add_trace(go.Histogram(accruacy))

            # fig = go.Figure()
            # fig.add_trace(go.Bar(x = CATEGORIES, y=[dd_count, dn_count, wd_count, wn_count], text=[dd_count, dn_count, wd_count, wn_count], textposition='auto'))
            # fig.update_xaxes(title='ROAD')
            # fig.update_yaxes(title='Images')
            # fig.show()

        # if self.SHOW is not None:
        #     freq, _ = np.histogram(accruacy, bins=7, range=(30, 100))
        #     x = [f'{i} - {(i + 9)}' for i in range(30, 100, 10)]
        #     x[-1] = '90 - 100'
        #     fig = go.Figure()
        #     fig.add_trace(go.Bar(x=x, y=freq, text=freq, textposition='auto'))
        #     fig.update_xaxes(title='Acc')
        #     fig.update_yaxes(title='Images')
        #     fig.show()

            # return np.argmax(prediction_list), prediction_list


if __name__ == '__main__':
    SHOW_CATEGORIES = None
    # - Change it-
    model_path = r'D:\Harry\000.DataAnalysis\004.Model\53R\53R_FOG_cp_20210106165128.tf'
    model_path2 = r'D:\Harry\000.DataAnalysis\004.Model\2model_test\cp_20210204160921.tf'
    # folder_path = r'F:\MK-SD53R\2021-02-02\202'
    folder_path = r'F:\test\snow'
    SAMPLING_RATIO = 100
    CATEGORIES = ['normal', 'fog']
    CATEGORIES2 = ['normal', 'snow']
    # SHOW_CATEGORIES = 4
    ROI = '53R_202'  # 33A_201, 33A_201_old, 33A_201_fusion, 33C_201, 33C_201_fusion, 53R_201, 53R_202  || 201 = Road, 202 = Weather
    classifier = Classifier(MODEL_PATH1 = model_path, MODEL_PATH2 = model_path2, PATH = folder_path, CATEGORIES = CATEGORIES, CATEGORIES2 = CATEGORIES2, SAMPLING_RATIO = SAMPLING_RATIO)
    #
    # start_time = time.time()
    # if SHOW_CATEGORIES is None:
    #     classifier = Classifier(model_path, folder_path, CATEGORIES, SAMPLING_RATIO, ROI, MOVE=True)
    # if SHOW_CATEGORIES is not None:
    #     classifier = Classifier(model_path, folder_path, CATEGORIES, SAMPLING_RATIO, ROI, SHOW_CATEGORIES)

    classifier()
    # end_time = time.time()
    # elsapsed_time = (end_time - start_time)
    # elsapsed_time = str(datetime.timedelta(seconds=elsapsed_time)).split(".")
    # elsapsed_time = elsapsed_time[0]
    # start_str = str(datetime.timedelta(seconds=start_time)).split(".")
    # start_str = start_str[0]
    # end_str = str(datetime.timedelta(seconds=end_time)).split(".")
    # end_str = end_str[0]
    # print(f'Start Time : {start_str} ; End time : {end_str} ; Elapsed time : {elsapsed_time}')


    '''
    t1 = threading.Thread(target=classify_roadsurface_status,
                          args=('rtsp://rtsp2020:rtsp2020@192.168.1.201:554/cam/realmonitor?channel=1&subtype=0', ))

    classify_roadsurface_status('rtsp://rtsp2020:rtsp2020@192.168.1.201:554/cam/realmonitor?channel=1&subtype=0')
    '''
