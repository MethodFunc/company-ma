# dataset maker v0.20
import cv2
import fnmatch
import numpy as np
import os
import random
import shutil
import logging.config

logging.config.fileConfig('logging.conf')
logger = logging.getLogger('dataset_maker')

SOURCE_PATH = 'F:/DATA/33A(sample)/Day'
# SOURCE_PATH = 'D:/MKWS01/python/tf2trainer/33A(sample)/Day'
FRAME_WIDTH, FRAME_HEIGHT = 1520, 2688
ROI_WIDTH, ROI_HEIGHT = 150, 150
ROI = []
#ROI1 = [(2, 4), (3, 4), (1, 5), (2, 5), (3, 5), (0, 6), (1, 6), (2, 6),
#        (0, 7), (1, 7), (2, 7), (0, 8), (1, 8), (0, 9), (0, 10)]
#ROI2 = [(8, 6), (9, 7), (6, 8), (7, 8), (9, 8), (6, 9), (7, 9), (6, 10),
#        (7, 10), (8, 10), (6, 11), (7, 11), (8, 11), (6, 12), (7, 12), (8, 12), (9, 12),
#        (6, 13), (7, 13), (8, 13), (9, 13), (6, 14), (7, 14), (8, 14), (9, 14),
#        (6, 15), (7, 15), (8, 15), (9, 15), (6, 16), (7, 16), (8, 16), (9, 16)]
CATEGORIES = ['Dry', 'Wet']
SAMPLE_NUMBER = 300

def roi_mean(roi_img, roi):
    mean = [roi_img[i].mean() for i in range(len(roi_img))]
    mean = np.array(mean).reshape(-1, len(roi))

    return mean

def load_dataset(source_path, **kwargs):
    logger.info(f'=== Dataset Maker Start ===')
    frame_width = kwargs['frame_width'] if 'frame_width' in kwargs else 1520
    frame_height = kwargs['frame_height'] if 'frame_height' in kwargs else 2688
    roi_width = kwargs['roi_width'] if 'roi_width' in kwargs else 150
    roi_height = kwargs['roi_height'] if 'roi_height' in kwargs else 150
    roi_depth = kwargs['roi_depth'] if 'roi_depth' in kwargs else 3
    sample_number = kwargs['sample_number'] if 'sample_number' in kwargs else 1000
    validation_ratio = kwargs['validation_ratio'] if 'validation_ratio' in kwargs else 0.3
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

    for img_file, img_label in train_set:
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
    for img_file, img_label in test_set:
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
