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
#import logging.config

#logging.config.fileConfig('logging.conf')
#logger = logging.getLogger('rsclassifier')
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)
'''
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
tf.keras.backend.set_session(tf.Session(config=config))
'''

model_day = tf.keras.models.load_model('D:/MKWS01/datasets/53R(Sunrise_Sunset)/cp_20201209112107.tf')

ROI_WIDTH, ROI_HEIGHT = 150, 150
CATEGORIES = ['건조', '습윤']


def predict_roadsurface(**kwargs):
    global model_day, model_night, ROI_WIDTH, ROI_HEIGHT, ROI, prediction_list

    path = 'D:/MKWS01/datasets/53R(Sunrise_Sunset)/test3(Nor)/wet/'
    path2 = 'C:/Users/ThreadRipper/Desktop/dry&wet_train/color_mode_33A_data/20200910_Dry,Wet_data/day+night/other(33A_S43)_test_image/'


    images = glob.glob(f'{path}/*.jpg')
    count = 1

    for imgs in images:
        # image_frame = cv2.imread(f'{path}/CAM201_20200710001144688488_0.jpg')
        # image_frame = np.array(image_frame).reshape(-1, 1520, 2688, 3)
        # image_frame = image_frame / 255.

        is_daytime = 1
        # current_time = current_time.hour * 10000 + current_time.minute * 100 + current_time.second
        '''
        if (current_time > SUNRISE) and (current_time < SUNSET):
            is_daytime = 1
        number_of_categories = 5 if is_daytime else 2
        '''

        filedir = os.path.basename(imgs)
        filename = os.path.splitext(filedir)
        img = cv2.imread(imgs)

        imgss = np.array(img).reshape(-1, 150, 150, 3)
        imgsss = imgss / 255.0
        pred_values = model_day.predict(imgsss)#  if is_daytime else list(model_night.predict(roi_set))
        # print(np.int64(pred_values*10))
        preds = []
        for pred in pred_values:
            preds.append(np.argmax(pred))

        prediction_list = []
        for i in range(5):
            prediction_list.append(preds.count(i))
        '''if prediction_list[0] == prediction_list[1]:
            prediction_list[1] += 1'''
        #print('filename :', filename[0], 'pred :', preds, 'count :', prediction_list, 'class :', np.argmax(prediction_list), 'category :', CATEGORIES[np.argmax(prediction_list)])
        if np.argmax(prediction_list) == 0:
            # print(f'filename : {filename[0]}, count : {prediction_list}, class : {np.argmax(prediction_list)}')
            print(f'filename : {filename[0]}, count : {np.int64(pred_values*100)[0]}, class : {CATEGORIES[np.argmax(prediction_list)]}')
            print(f'error_num : {count}, Total_num : {len(images)}, acc :{(len(images) - count)/len(images)*100}%')
            count += 1
        '''    cv2.imwrite(f'{path2}/dry/{filename[0]}_{np.argmax(prediction_list)}.jpg', img)
        else:
            cv2.imwrite(f'{path2}/wet/{filename[0]}_{np.argmax(prediction_list)}.jpg', img)'''
    return np.argmax(prediction_list), prediction_list


if __name__ == '__main__':
    '''
    t1 = threading.Thread(target=classify_roadsurface_status,
                          args=('rtsp://rtsp2020:rtsp2020@192.168.1.201:554/cam/realmonitor?channel=1&subtype=0', ))

    classify_roadsurface_status('rtsp://rtsp2020:rtsp2020@192.168.1.201:554/cam/realmonitor?channel=1&subtype=0')
    '''
    predict_roadsurface()
