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

SOURCE_PATH = 'F:/DATA/33A(sample)/Day'
# SOURCE_PATH = 'D:/MKWS01/python/tf2trainer/33A(sample)/Day'
FRAME_WIDTH, FRAME_HEIGHT = 1520, 2688
ROI_WIDTH, ROI_HEIGHT = 150, 150
ROI = []
CATEGORIES = ['Dry', 'Wet']
SAMPLE_NUMBER = 300


def load_dataset(source_path, **kwargs):
    logger.info(f'=== Dataset Maker Start ===')
    roi_width = kwargs['roi_width'] if 'roi_width' in kwargs else 150
    roi_height = kwargs['roi_height'] if 'roi_height' in kwargs else 150
    roi_depth = kwargs['roi_depth'] if 'roi_depth' in kwargs else 3
    sample_number = kwargs['sample_number'] if 'sample_number' in kwargs else 1000
    validation_ratio = kwargs['validation_ratio'] if 'validation_ratio' in kwargs else 0.2

    if 'roi' in kwargs:
        roi = kwargs['roi']
    else:
        # 수정 필요
        roi = []

    if 'categories' in kwargs:
        categories = kwargs['categories']
    else:
        # 수정 필요
        categories = ['Dry', 'Wet']

    for cat_item in categories:
        path = ''.join([str(source_path).replace("\\", "/"), "/", str(cat_item)])
        image_file_list = fnmatch.filter(os.listdir(path), '*.jpg')
        sample_number = min(len(image_file_list), sample_number)
        if sample_number == 0:
            logger.error(f'No image file exists in source path: {path}.')
            exit()
        elif len(image_file_list) < sample_number:
            logger.warning(f'Number of {cat_item} image ({len(image_file_list)}) is smaller'
                           f' than SAMPLE NUMBER({sample_number}).')

    train_set, test_set = [], []
    for cat_item in categories:
        path = ''.join([str(source_path).replace("\\", "/"), "/", str(cat_item)])
        image_file_list = fnmatch.filter(os.listdir(path), '*.jpg')
        random.shuffle(image_file_list)

        test_number = round(sample_number * validation_ratio)
        train_number = sample_number - test_number
        label = categories.index(cat_item)

        for img_file in image_file_list[:train_number]:
            train_set.append([img_file, label])
        for img_file in image_file_list[train_number:sample_number]:
            test_set.append([img_file, label])

    random.shuffle(train_set)
    random.shuffle(test_set)

    train_image, test_image, train_label, test_label = [], [], [], []
    logger.info(f'Processing training dataset...')

    for img_file, img_label in tqdm(train_set):
        try:
            path = ''.join([str(source_path).replace("\\", "/"), "/", str(categories[img_label])])
            img_file_path = ''.join([str(path), "/", str(img_file)])
            img = cv2.imread(img_file_path)
            for (i, j) in roi:
                x, y = i * roi_width, j * roi_height
                roi_img = img[y:y + roi_height, x:x + roi_width]
                train_image.append(roi_img)
                train_label.append(img_label)
        except Exception as err:
            logger.error(f'image processing failed at {str(img_file)}')
            pass

    logger.info(f'Processing test dataset...')
    for img_file, img_label in tqdm(test_set):
        try:
            path = ''.join([str(source_path).replace("\\", "/"), "/", str(categories[img_label])])
            img_file_path = ''.join([str(path), "/", str(img_file)])
            img = cv2.imread(img_file_path)
            for (i, j) in roi:
                x, y = i * roi_width, j * roi_height
                roi_img = img[y:y + roi_height, x:x + roi_width]
                test_image.append(roi_img)
                test_label.append(img_label)
        except Exception as err:
            logger.error(f'image processing failed at {str(img_file)}')
            pass

    train_image = np.array(train_image).reshape(-1, roi_height, roi_width, roi_depth)
    train_label = np.array(train_label)
    test_image = np.array(test_image).reshape(-1, roi_height, roi_width, roi_depth)
    test_label = np.array(test_label)
    logger.info(f'Dataset Processing Finished.')
    logger.info(f'train_image: {train_image.shape}, train_label: {train_label.shape}')
    logger.info(f'test_image: {test_image.shape}, test_label: {test_label.shape}')
    return (train_image, train_label), (test_image, test_label)


if __name__ == '__main__':
    load_dataset(source_path=SOURCE_PATH, roi=ROI1, sample_number=SAMPLE_NUMBER)
