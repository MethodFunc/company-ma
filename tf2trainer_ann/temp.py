# dataset maker v0.20
import cv2
import fnmatch
import numpy as np
import os
import random
import shutil
import logging.config
from tqdm import tqdm

logging.config.fileConfig('logging.conf')
logger = logging.getLogger('dataset_maker')

SOURCE_PATH = 'F:/MK-SD53R/2021-02-04/202'
# SOURCE_PATH = 'D:/MKWS01/python/tf2trainer/33A(sample)/Day'
FRAME_WIDTH, FRAME_HEIGHT = 1520, 2688
ROI_WIDTH, ROI_HEIGHT = 150, 150
roi = [(2, 4), (3, 4), (1, 5), (2, 5), (3, 5), (0, 6), (1, 6), (2, 6),
       (0, 7), (1, 7), (2, 7), (0, 8), (1, 8), (0, 9), (0, 10)]
#ROI2 = [(8, 6), (9, 7), (6, 8), (7, 8), (9, 8), (6, 9), (7, 9), (6, 10),
#        (7, 10), (8, 10), (6, 11), (7, 11), (8, 11), (6, 12), (7, 12), (8, 12), (9, 12),
#        (6, 13), (7, 13), (8, 13), (9, 13), (6, 14), (7, 14), (8, 14), (9, 14),
#        (6, 15), (7, 15), (8, 15), (9, 15), (6, 16), (7, 16), (8, 16), (9, 16)]
CATEGORIES = ['normal_day', 'normal_night']
SAMPLE_NUMBER = 250
validation_ratio = 0.2

for cat_item in CATEGORIES:
        path = ''.join([str(SOURCE_PATH).replace("\\", "/"), "/", str(cat_item)])
        image_file_list = fnmatch.filter(os.listdir(path), '*.jpg')
        SAMPLE_NUMBER = min(len(image_file_list), SAMPLE_NUMBER)
        if SAMPLE_NUMBER == 0:
            logger.error(f'No image file exists in source path: {path}.')
            exit()
        elif len(image_file_list) < SAMPLE_NUMBER:
            logger.warning(f'Number of {cat_item} image ({len(image_file_list)}) is smaller'
                           f' than SAMPLE NUMBER({SAMPLE_NUMBER}).')


train_set, test_set = [], []

for cat_item in CATEGORIES:
    path = ''.join([str(SOURCE_PATH).replace("\\", "/"), "/", str(cat_item)])
    image_file_list = fnmatch.filter(os.listdir(path), '*.jpg')
    random.shuffle(image_file_list)

    test_number = round(SAMPLE_NUMBER * validation_ratio)
    train_number = SAMPLE_NUMBER - test_number
    label = CATEGORIES.index(cat_item)

    for img_file in image_file_list[:train_number]:
        train_set.append([img_file, label])
    for img_file in image_file_list[train_number:SAMPLE_NUMBER]:
        test_set.append([img_file, label])

random.shuffle(train_set)
random.shuffle(test_set)

logger.info(f'Processing training dataset...')
train_image, test_image, train_label, test_label = [], [], [], []
for img_file, img_label in tqdm(train_set):
    try:
        path = ''.join([str(SOURCE_PATH).replace("\\", "/"), "/", str(CATEGORIES[img_label])])
        img_file_path = ''.join([str(path), "/", str(img_file)])
        img = cv2.imread(img_file_path)
        for (i, j) in roi:
            x, y = i * ROI_WIDTH, j * ROI_HEIGHT
            roi_img = img[y:y + ROI_HEIGHT, x:x + ROI_WIDTH]
            train_image.append(roi_img)
            train_label.append(img_label)
    except Exception as err:
        logger.error(f'image processing failed at {str(img_file)}')
        pass

logger.info(f'Processing test dataset...')

for img_file, img_label in tqdm(test_set):
    try:
        path = ''.join([str(SOURCE_PATH).replace("\\", "/"), "/", str(CATEGORIES[img_label])])
        img_file_path = ''.join([str(path), "/", str(img_file)])
        img = cv2.imread(img_file_path)
        for (i, j) in roi:
            x, y = i * ROI_WIDTH, j * ROI_HEIGHT
            roi_img = img[y:y + ROI_HEIGHT, x:x + ROI_WIDTH]
            test_image.append(roi_img)
            test_label.append(img_label)
    except Exception as err:
        logger.error(f'image processing failed at {str(img_file)}')
        pass


