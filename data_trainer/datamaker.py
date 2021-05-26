import cv2
import os
import fnmatch
import logging.config
import random
import numpy as np


def __log():
    logging.config.fileConfig("logger.conf")
    logger = logging.getLogger("datamaker")

    return logger


def __check_category(source_path, categories):
    logger = __log()
    logger.info(f"Check category...")
    check_cat = {}
    for cat in categories:
        dir_path = f"{source_path}/{cat}"

        if not os.path.isdir(dir_path):
            logger.warning(f"Category: [{cat}] is not exist. Check your category or folder name.")

            check_cat[cat] = 1

    if check_cat:
        exit()
    logger.info("Check category Done")


def __check_image_count(source_path, categories, sample_image):
    logger = __log()
    logger.info(f"Check number of images...")
    check_log = []
    for cat in categories:
        dir_path = f"{source_path}/{cat}"
        img_file_list = fnmatch.filter(os.listdir(dir_path), "*.jpg")

        if len(img_file_list) == 0:
            logger.warning(f"{cat} image file does not exist.")
            logger.warning("Check your files")
            check_log.append(1)

        if len(img_file_list) < sample_image:
            logger.warning(f"{cat} image file less then sample_image.")
            logger.warning("Check your files")
            check_log.append(1)

    if 1 in check_log:
        exit()

    logger.info("Check number of images Done")


def split_image(source_path, categories, sample_image, validation_ratio):
    logger = __log()

    __check_category(source_path, categories)
    __check_image_count(source_path, categories, sample_image)

    logger.info(f"Start split images")
    train_set, test_set = [], []
    test_num = int(sample_image * validation_ratio)
    train_num = sample_image - test_num

    for cat in categories:
        dir_path = f"{source_path}/{cat}"
        file_list = fnmatch.filter(os.listdir(dir_path), "*.jpg")

        random.shuffle(file_list)
        label = categories.index(cat)

        for img in file_list[:train_num]:
            path = f"{dir_path}/{img}"
            train_set.append([path, label])

        for img in file_list[train_num:sample_image]:
            path = f"{dir_path}/{img}"
            test_set.append([path, label])

    random.shuffle(train_set)
    random.shuffle(test_set)

    return train_set, test_set


def load_data(dataset, roi, height, width, depth):
    images, labels = [], []
    for imgs, label in dataset:
        img = cv2.imread(imgs)
        for (i, j) in roi:
            x, y = i + width, j + height
            roi_img = img[y:y + height, x:x + width]

            images.append(roi_img)
            labels.append(label)

    images = np.array(images).reshape(-1, height, width, depth)
    labels = np.array(labels)

    return images, labels
